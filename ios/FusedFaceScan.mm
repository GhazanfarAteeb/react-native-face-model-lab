//
//  FusedFaceScan.mm — Fused native ANE scan pipeline for FaceModelLab.
//
//  Combines face detection (Vision) + crop/align + CoreML embedding into a single
//  native call per photo. This matches how rnbaby/BabyArt achieve their fastest
//  scanning: ~70ms per photo vs ~1,300ms on the JS→ORT CPU path.
//
//  Ported from rnbaby/ios/BabyRN/CoreMLFaceEmbedder.swift, adapted for FaceModelLab.
//
//  Supports two model input types:
//    - Image type (e.g. SphereFace): pass cropped CVPixelBuffer directly to CoreML
//    - MultiArray type (e.g. MobileFaceNet): manual BGRA→NCHW Float32 conversion
//

#import <React/RCTBridgeModule.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <Accelerate/Accelerate.h>

@interface NativeFusedFaceScan : NSObject <RCTBridgeModule>
@end

@implementation NativeFusedFaceScan {
  MLModel *_model;
  NSString *_inputName;
  NSString *_outputName;
  BOOL _inputIsImage;
  int _modelInputWidth;
  int _modelInputHeight;
}

RCT_EXPORT_MODULE(NativeFusedFaceScan);

+ (BOOL)requiresMainQueueSetup { return NO; }

- (dispatch_queue_t)methodQueue {
  return dispatch_queue_create("com.facemodellab.fusedscan", DISPATCH_QUEUE_SERIAL);
}

// ─── Model Loading ──────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(loadModel:(NSString *)assetName
                  inputName:(NSString *)inputName
                  outputName:(NSString *)outputName
                  inputWidth:(double)inputWidth
                  inputHeight:(double)inputHeight
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  @try {
    NSString *base = [assetName stringByDeletingPathExtension];
    NSURL *url = [[NSBundle mainBundle] URLForResource:base withExtension:@"mlmodelc"];
    if (!url) {
      resolve(@NO);
      return;
    }

    MLModelConfiguration *cfg = [MLModelConfiguration new];
    if (@available(iOS 16.0, *)) {
      cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    } else {
      cfg.computeUnits = MLComputeUnitsAll;
    }

    NSError *err = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!model || err) {
      resolve(@NO);
      return;
    }

    NSDictionary *ins = model.modelDescription.inputDescriptionsByName;
    NSDictionary *outs = model.modelDescription.outputDescriptionsByName;
    NSString *inName = (inputName.length > 0 && ins[inputName]) ? inputName : ins.allKeys.firstObject;
    NSString *outName = (outputName.length > 0 && outs[outputName]) ? outputName : outs.allKeys.firstObject;
    if (!inName || !outName) {
      resolve(@NO);
      return;
    }

    MLFeatureDescription *inDesc = ins[inName];
    _inputIsImage = (inDesc.type == MLFeatureTypeImage);

    _model = model;
    _inputName = inName;
    _outputName = outName;
    _modelInputWidth = (int)inputWidth;
    _modelInputHeight = (int)inputHeight;

    NSLog(@"[FusedFaceScan] Loaded %@ | input=%@ (%@ %dx%d) | output=%@",
          base, inName, _inputIsImage ? @"Image" : @"MultiArray",
          _modelInputWidth, _modelInputHeight, outName);

    resolve(@YES);
  } @catch (NSException *e) {
    resolve(@NO);
  }
}

// ─── Fused scanPhoto ───────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(scanPhoto:(NSString *)imagePath
                  paddingRatio:(double)paddingRatio
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  if (!_model) {
    reject(@"no_model", @"Call loadModel first", nil);
    return;
  }

  @try {
    // 1. Decode image
    NSDate *t0 = [NSDate date];
    NSString *path = imagePath;
    if ([path hasPrefix:@"file://"]) {
      path = [path substringFromIndex:7];
    }
    UIImage *image = [UIImage imageWithContentsOfFile:path];
    if (!image) {
      reject(@"decode_error", @"Could not load image", nil);
      return;
    }
    image = [self fixOrientation:image];
    CGImageRef cgImage = image.CGImage;
    if (!cgImage) {
      reject(@"decode_error", @"Could not create CGImage", nil);
      return;
    }
    size_t imgW = CGImageGetWidth(cgImage);
    size_t imgH = CGImageGetHeight(cgImage);
    NSTimeInterval decodeMs = [[NSDate date] timeIntervalSinceDate:t0] * 1000;

    // 2. Face detection (Vision)
    NSDate *t1 = [NSDate date];
    NSArray *faces = [self detectFaces:cgImage imgW:imgW imgH:imgH];
    NSTimeInterval detectMs = [[NSDate date] timeIntervalSinceDate:t1] * 1000;

    if (faces.count == 0) {
      resolve(@{
        @"faces": @[],
        @"embeddings": @[],
        @"imageWidth": @(imgW),
        @"imageHeight": @(imgH),
        @"timings": @{
          @"decodeMs": @(decodeMs),
          @"detectMs": @(detectMs),
          @"cropMs": @(0),
          @"alignMs": @(0),
          @"embedMs": @(0),
        },
      });
      return;
    }

    // 3. For each face: crop + align + embed
    NSTimeInterval totalCropMs = 0;
    NSTimeInterval totalEmbedMs = 0;
    NSMutableArray *faceResults = [NSMutableArray array];
    NSMutableArray *embeddings = [NSMutableArray array];

    for (NSDictionary *face in faces) {
      @autoreleasepool {
        double left = [face[@"left"] doubleValue];
        double top = [face[@"top"] doubleValue];
        double w = [face[@"width"] doubleValue];
        double h = [face[@"height"] doubleValue];
        NSArray *alignPts = face[@"alignPts"];

        NSDate *tCrop = [NSDate date];
        CGImageRef cropImg = [self cropAndAlign:cgImage
                                          left:left top:top width:w height:h
                                     alignPts:alignPts
                                         imgW:imgW imgH:imgH
                                     padding:paddingRatio];
        NSTimeInterval cropMs = [[NSDate date] timeIntervalSinceDate:tCrop] * 1000;
        totalCropMs += cropMs;

        if (!cropImg) continue;

        NSDate *tEmbed = [NSDate date];
        NSArray *embedding = [self embedCrop:cropImg];
        NSTimeInterval embedMs = [[NSDate date] timeIntervalSinceDate:tEmbed] * 1000;
        totalEmbedMs += embedMs;

        CGImageRelease(cropImg);

        if (embedding && embedding.count > 0) {
          [faceResults addObject:@{
            @"left": @(left),
            @"top": @(top),
            @"width": @(w),
            @"height": @(h),
          }];
          [embeddings addObject:embedding];
        }
      }
    }

    resolve(@{
      @"faces": faceResults,
      @"embeddings": embeddings,
      @"imageWidth": @(imgW),
      @"imageHeight": @(imgH),
      @"timings": @{
        @"decodeMs": @(decodeMs),
        @"detectMs": @(detectMs),
        @"cropMs": @(totalCropMs),
        @"alignMs": @(0),
        @"embedMs": @(totalEmbedMs),
      },
    });
  } @catch (NSException *e) {
    reject(@"exception", e.reason, nil);
  }
}

// ─── Release ────────────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(release:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  _model = nil;
  _inputName = nil;
  _outputName = nil;
  resolve(nil);
}

// ─── Device Info ────────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(getDeviceInfo:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  NSMutableDictionary *info = [NSMutableDictionary new];
  if (@available(iOS 16.0, *)) {
    info[@"aneAvailable"] = @YES;
  } else {
    info[@"aneAvailable"] = @NO;
  }
  info[@"systemName"] = [UIDevice currentDevice].systemName;
  info[@"systemVersion"] = [UIDevice currentDevice].systemVersion;
  info[@"model"] = [UIDevice currentDevice].model;
  resolve(info);
}

// ─── Private: Face Detection (Vision) ──────────────────────────────────────────
// Matches BabyArt/rnbaby: VNDetectFaceLandmarksRequest revision 3, detect on
// downscaled copy (max 1280px), coordinates scaled back to full-res.

- (NSArray *)detectFaces:(CGImageRef)cgImage imgW:(size_t)imgW imgH:(size_t)imgH {
  // Downscale for detection (matching BabyArt's detectionMaxEdge = 1280)
  static const CGFloat kDetectionMaxEdge = 1280;
  CGImageRef detectImage = cgImage;
  CGFloat detectW = imgW, detectH = imgH;
  CGFloat longEdge = MAX(imgW, imgH);
  if (longEdge > kDetectionMaxEdge) {
    CGFloat scale = kDetectionMaxEdge / longEdge;
    size_t scaledW = (size_t)(imgW * scale);
    size_t scaledH = (size_t)(imgH * scale);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(NULL, scaledW, scaledH, 8, scaledW * 4,
                                              cs, kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(cs);
    if (ctx) {
      CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
      CGRect dstRect = CGRectMake(0, 0, scaledW, scaledH);
      CGContextDrawImage(ctx, dstRect, cgImage);
      detectImage = CGBitmapContextCreateImage(ctx);
      CGContextRelease(ctx);
      detectW = scaledW;
      detectH = scaledH;
    }
  }

  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block NSArray *results = @[];

  VNDetectFaceLandmarksRequest *req = [[VNDetectFaceLandmarksRequest alloc]
    initWithCompletionHandler:^(VNRequest *request, NSError *error) {
      if (error || !request.results) {
        dispatch_semaphore_signal(sem);
        return;
      }
      NSMutableArray *faces = [NSMutableArray array];
      // Scale factors: detection coords → full-res pixel coords
      CGFloat sx = imgW / detectW;
      CGFloat sy = imgH / detectH;

      for (VNFaceObservation *obs in request.results) {
        CGRect bbox = obs.boundingBox;
        // Vision normalized bottom-left → full-res top-left pixels
        double left = bbox.origin.x * imgW;
        double top = (1.0 - bbox.origin.y - bbox.size.height) * imgH;
        double w = bbox.size.width * imgW;
        double h = bbox.size.height * imgH;

        if (MIN(w, h) < 20) continue;

        NSArray *alignPts = nil;
        VNFaceLandmarks2D *landmarks = obs.landmarks;
        if (landmarks) {
          // Extract landmarks in detection-res coords, then scale to full-res
          alignPts = [self extractAlignmentPoints:landmarks
                                           imgW:(size_t)detectW imgH:(size_t)detectH
                                             sx:sx sy:sy];
        }

        [faces addObject:@{
          @"left": @(left),
          @"top": @(top),
          @"width": @(w),
          @"height": @(h),
          @"alignPts": alignPts ?: @[],
        }];
      }
      results = [faces copy];
      dispatch_semaphore_signal(sem);
    }];

  VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:detectImage options:@{}];
  NSError *err = nil;
  [handler performRequests:@[req] error:&err];
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

  // Release downscaled detection image if we created one
  if (detectImage != cgImage) {
    CGImageRelease(detectImage);
  }

  return results;
}

// ─── Private: Extract 5-point ArcFace alignment points ─────────────────────────

// Extract 5-point ArcFace alignment points from Vision landmarks.
// Landmarks are in detection-resolution coords; sx/sy scale them to full-res pixels.
// Y-flip: Vision's pointsInImageOfSize returns bottom-left origin, we need top-left.
- (NSArray *)extractAlignmentPoints:(VNFaceLandmarks2D *)landmarks
                             imgW:(size_t)imgW imgH:(size_t)imgH
                               sx:(CGFloat)sx sy:(CGFloat)sy {
  // Helper: compute centroid of a landmark region, Y-flip, and scale to full-res
  CGPoint (^centroidScaled)(VNFaceLandmarkRegion2D *) = ^CGPoint(VNFaceLandmarkRegion2D *region) {
    if (!region || region.pointCount == 0) return CGPointMake(-1, -1);
    const CGPoint *pts = [region pointsInImageOfSize:CGSizeMake(imgW, imgH)];
    if (!pts) return CGPointMake(-1, -1);
    CGFloat sumX = 0, sumY = 0;
    NSUInteger count = region.pointCount;
    for (NSUInteger i = 0; i < count; i++) {
      sumX += pts[i].x;
      sumY += pts[i].y;
    }
    // Y-flip (bottom-left → top-left) and scale to full-res
    return CGPointMake((sumX / count) * sx, (imgH - sumY / count) * sy);
  };

  // Prefer pupil centroid, fall back to eye region centroid (matching BabyArt)
  CGPoint leftEye = centroidScaled(landmarks.leftPupil);
  if (leftEye.x < 0) leftEye = centroidScaled(landmarks.leftEye);
  CGPoint rightEye = centroidScaled(landmarks.rightPupil);
  if (rightEye.x < 0) rightEye = centroidScaled(landmarks.rightEye);
  CGPoint nose = centroidScaled(landmarks.nose);

  // Mouth corners: x-extremes of outerLips, Y-flipped and scaled (matching BabyArt)
  CGPoint mouthLeft = CGPointMake(-1, -1);
  CGPoint mouthRight = CGPointMake(-1, -1);

  if (landmarks.outerLips && landmarks.outerLips.pointCount >= 2) {
    const CGPoint *pts = [landmarks.outerLips pointsInImageOfSize:CGSizeMake(imgW, imgH)];
    NSUInteger count = landmarks.outerLips.pointCount;
    CGFloat loX = pts[0].x, loY = pts[0].y;
    CGFloat hiX = pts[0].x, hiY = pts[0].y;
    for (NSUInteger i = 1; i < count; i++) {
      if (pts[i].x < loX) { loX = pts[i].x; loY = pts[i].y; }
      if (pts[i].x > hiX) { hiX = pts[i].x; hiY = pts[i].y; }
    }
    // Y-flip and scale to full-res
    mouthLeft = CGPointMake(loX * sx, (imgH - loY) * sy);
    mouthRight = CGPointMake(hiX * sx, (imgH - hiY) * sy);
  }

  if (leftEye.x < 0 || rightEye.x < 0 || nose.x < 0 || mouthLeft.x < 0 || mouthRight.x < 0) {
    return nil;
  }

  return @[
    @[@(leftEye.x), @(leftEye.y)],
    @[@(rightEye.x), @(rightEye.y)],
    @[@(nose.x), @(nose.y)],
    @[@(mouthLeft.x), @(mouthLeft.y)],
    @[@(mouthRight.x), @(mouthRight.y)],
  ];
}

// ─── Private: Crop + Align ─────────────────────────────────────────────────────

- (CGImageRef)cropAndAlign:(CGImageRef)srcImage
                      left:(double)left top:(double)top width:(double)w height:(double)h
                 alignPts:(NSArray *)alignPts
                     imgW:(size_t)imgW imgH:(size_t)imgH
                 padding:(double)padding {
  if (alignPts.count == 5) {
    return [self arcfaceWarp:srcImage alignPts:alignPts imgW:imgW imgH:imgH];
  }
  return [self bboxCrop:srcImage left:left top:top width:w height:h
                  imgW:imgW imgH:imgH padding:padding];
}

// ─── Private: ArcFace Similarity Warp ──────────────────────────────────────────

- (CGImageRef)arcfaceWarp:(CGImageRef)srcImage
                  alignPts:(NSArray *)alignPts
                      imgW:(size_t)imgW imgH:(size_t)imgH {
  CGPoint srcPts[5];
  for (int i = 0; i < 5; i++) {
    NSArray *pt = alignPts[i];
    srcPts[i] = CGPointMake([pt[0] doubleValue], [pt[1] doubleValue]);
  }

  // Use the model's actual input dimensions (set during loadModel)
  int outW = _modelInputWidth > 0 ? _modelInputWidth : 112;
  int outH = _modelInputHeight > 0 ? _modelInputHeight : 112;

  // ArcFace canonical 5-point template, scaled to outW × outH
  CGPoint dstPts[5];
  double templateCoords[5][2] = {
    {38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366},
    {41.5493, 92.3655}, {70.7299, 92.2041},
  };
  for (int i = 0; i < 5; i++) {
    dstPts[i] = CGPointMake(templateCoords[i][0] * outW / 112.0,
                             templateCoords[i][1] * outH / 112.0);
  }

  // Solve similarity transform
  double a, b, tx, ty;
  if (![self solveSimilaritySrc:srcPts dst:dstPts a:&a b:&b tx:&tx ty:&ty]) {
    return NULL;
  }

  // Apply transform via CGContext
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  CGContextRef ctx = CGBitmapContextCreate(NULL, outW, outH, 8, outW * 4,
                                            colorSpace, kCGImageAlphaPremultipliedLast);
  CGColorSpaceRelease(colorSpace);
  if (!ctx) return NULL;

  CGContextSetRGBFillColor(ctx, 0, 0, 0, 1);
  CGContextFillRect(ctx, CGRectMake(0, 0, outW, outH));

  // Build CGAffineTransform: flip Y (CGContext is bottom-left), then apply similarity
  CGAffineTransform flip = CGAffineTransformMake(1, 0, 0, -1, 0, outH);
  CGAffineTransform sim = CGAffineTransformMake(a, b, -b, a, tx, ty);
  CGAffineTransform combined = CGAffineTransformConcat(sim, flip);
  CGContextConcatCTM(ctx, combined);

  // Draw source image at its original size
  CGRect srcRect = CGRectMake(0, 0, imgW, imgH);
  CGContextDrawImage(ctx, srcRect, srcImage);

  CGImageRef result = CGBitmapContextCreateImage(ctx);
  CGContextRelease(ctx);
  return result;
}

// ─── Private: Solve similarity transform via least squares ─────────────────────

- (BOOL)solveSimilaritySrc:(CGPoint[5])src dst:(CGPoint[5])dst
                         a:(double *)a b:(double *)b tx:(double *)tx ty:(double *)ty {
  double ATA[4][4] = {{0}};
  double ATb[4] = {0};

  for (int i = 0; i < 5; i++) {
    double x = src[i].x, y = src[i].y;
    double dx = dst[i].x, dy = dst[i].y;

    double r1[4] = {x, -y, 1, 0};
    for (int j = 0; j < 4; j++) {
      ATb[j] += r1[j] * dx;
      for (int k = 0; k < 4; k++) ATA[j][k] += r1[j] * r1[k];
    }

    double r2[4] = {y, x, 0, 1};
    for (int j = 0; j < 4; j++) {
      ATb[j] += r2[j] * dy;
      for (int k = 0; k < 4; k++) ATA[j][k] += r2[j] * r2[k];
    }
  }

  double aug[4][5];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) aug[i][j] = ATA[i][j];
    aug[i][4] = ATb[i];
  }

  for (int col = 0; col < 4; col++) {
    int pivot = col;
    double maxAbs = fabs(aug[col][col]);
    for (int r = col + 1; r < 4; r++) {
      if (fabs(aug[r][col]) > maxAbs) {
        maxAbs = fabs(aug[r][col]);
        pivot = r;
      }
    }
    if (maxAbs < 1e-12) return NO;
    if (pivot != col) {
      for (int j = 0; j < 5; j++) {
        double tmp = aug[col][j]; aug[col][j] = aug[pivot][j]; aug[pivot][j] = tmp;
      }
    }
    for (int r = 0; r < 4; r++) {
      if (r == col) continue;
      double factor = aug[r][col] / aug[col][col];
      for (int j = col; j < 5; j++) aug[r][j] -= factor * aug[col][j];
    }
  }

  double result[4];
  for (int i = 0; i < 4; i++) result[i] = aug[i][4] / aug[i][i];

  *a = result[0]; *b = result[1]; *tx = result[2]; *ty = result[3];
  return YES;
}

// ─── Private: BBox center-square crop (fallback) ───────────────────────────────

- (CGImageRef)bboxCrop:(CGImageRef)srcImage
                  left:(double)left top:(double)top width:(double)w height:(double)h
                  imgW:(size_t)imgW imgH:(size_t)imgH padding:(double)padding {
  double padW = w * padding;
  double padH = h * padding;
  double cropL = MAX(0, left - padW);
  double cropT = MAX(0, top - padH);
  double cropW = MIN(w + 2 * padW, (double)(imgW - cropL));
  double cropH = MIN(h + 2 * padH, (double)(imgH - cropT));

  if (cropW < 10 || cropH < 10) return NULL;

  double side = MIN(cropW, cropH);
  double cx = cropL + cropW / 2;
  double cy = cropT + cropH / 2;
  double sqL = MAX(0, cx - side / 2);
  double sqT = MAX(0, cy - side / 2);
  sqL = MIN(sqL, (double)(imgW - side));
  sqT = MIN(sqT, (double)(imgH - side));

  CGRect cropRect = CGRectMake(sqL, sqT, side, side);
  CGImageRef cropped = CGImageCreateWithImageInRect(srcImage, cropRect);
  if (!cropped) return NULL;

  // Use the model's actual input dimensions (set during loadModel)
  int outW = _modelInputWidth > 0 ? _modelInputWidth : 112;
  int outH = _modelInputHeight > 0 ? _modelInputHeight : 112;

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  CGContextRef ctx = CGBitmapContextCreate(NULL, outW, outH, 8, outW * 4,
                                            colorSpace, kCGImageAlphaPremultipliedLast);
  CGColorSpaceRelease(colorSpace);
  if (!ctx) { CGImageRelease(cropped); return NULL; }

  CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
  CGRect dstRect = CGRectMake(0, 0, outW, outH);
  CGRect srcRect = CGRectMake(0, 0, CGImageGetWidth(cropped), CGImageGetHeight(cropped));
  CGContextDrawImage(ctx, dstRect, cropped);
  CGImageRelease(cropped);

  CGImageRef result = CGBitmapContextCreateImage(ctx);
  CGContextRelease(ctx);
  return result;
}

// ─── Private: CoreML Embedding ─────────────────────────────────────────────────

- (NSArray *)embedCrop:(CGImageRef)cropImage {
  if (!_model) return nil;

  NSError *err = nil;
  MLFeatureValue *inputFV;

  if (_inputIsImage) {
    // Image type: convert CGImage → CVPixelBuffer → MLFeatureValue
    CVPixelBufferRef pixelBuffer = [self pixelBufferFromCGImage:cropImage];
    if (!pixelBuffer) return nil;
    inputFV = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
    CVPixelBufferRelease(pixelBuffer);
  } else {
    // MultiArray type: convert CGImage → NCHW Float32 tensor
    inputFV = [self cgImageToMultiArray:cropImage];
    if (!inputFV) return nil;
  }

  MLDictionaryFeatureProvider *fp =
    [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{_inputName : inputFV}
                                                        error:&err];
  if (!fp || err) return nil;

  id<MLFeatureProvider> out = [_model predictionFromFeatures:fp error:&err];
  if (!out || err) return nil;

  MLMultiArray *res = [out featureValueForName:_outputName].multiArrayValue;
  if (!res) return nil;

  // Read output
  NSMutableArray *vec = [NSMutableArray arrayWithCapacity:res.count];
  NSInteger dataType = res.dataType;

  if (dataType == 16) {
    // Float16 — read via raw UInt16 bit decode
    uint16_t *ptr = (uint16_t *)res.dataPointer;
    NSInteger count = res.count;
    for (NSInteger i = 0; i < count; i++) {
      uint16_t h = ptr[i];
      int s = (h >> 15) & 1;
      int e = (h >> 10) & 0x1F;
      int m = h & 0x3FF;
      float val;
      if (e == 0) {
        val = (s ? -1 : 1) * pow(2, -14) * (m / 1024.0f);
      } else if (e == 31) {
        val = (m == 0) ? (s ? -INFINITY : INFINITY) : NAN;
      } else {
        val = (s ? -1 : 1) * pow(2, e - 15) * (1 + m / 1024.0f);
      }
      [vec addObject:@(val)];
    }
  } else {
    // Float32
    NSInteger count = res.count;
    for (NSInteger i = 0; i < count; i++) {
      [vec addObject:@([res[i] floatValue])];
    }
  }

  return vec;
}

// ─── Private: CGImage → CVPixelBuffer (for Image input models) ─────────────────

- (CVPixelBufferRef)pixelBufferFromCGImage:(CGImageRef)cgImage {
  size_t width = CGImageGetWidth(cgImage);
  size_t height = CGImageGetHeight(cgImage);

  NSDictionary *attrs = @{
    (NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
    (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
  };

  CVPixelBufferRef pixelBuffer = NULL;
  CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width, height,
                                         kCVPixelFormatType_32BGRA,
                                         (__bridge CFDictionaryRef)attrs,
                                         &pixelBuffer);
  if (status != kCVReturnSuccess) return NULL;

  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
  void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
  size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  CGContextRef ctx = CGBitmapContextCreate(baseAddress, width, height, 8,
                                            bytesPerRow, colorSpace,
                                            kCGImageAlphaPremultipliedFirst);
  CGColorSpaceRelease(colorSpace);

  if (ctx) {
    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), cgImage);
    CGContextRelease(ctx);
  }

  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
  return pixelBuffer;
}

// ─── Private: CGImage → MLMultiArray (for MultiArray input models) ─────────────

- (MLFeatureValue *)cgImageToMultiArray:(CGImageRef)cgImage {
  MLFeatureDescription *inDesc = _model.modelDescription.inputDescriptionsByName[_inputName];
  NSArray<NSNumber *> *shape = inDesc.multiArrayConstraint.shape;
  if (!shape || shape.count < 4) return nil;

  int modelH = [[shape objectAtIndex:shape.count - 2] intValue];
  int modelW = [[shape lastObject] intValue];

  // Resize if needed
  CGImageRef resized = cgImage;
  BOOL didResize = NO;
  size_t w = CGImageGetWidth(cgImage);
  size_t h = CGImageGetHeight(cgImage);
  if (w != (size_t)modelW || h != (size_t)modelH) {
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(NULL, modelW, modelH, 8, modelW * 4,
                                              cs, kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(cs);
    if (ctx) {
      CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
      CGRect dstRect = CGRectMake(0, 0, modelW, modelH);
      CGContextDrawImage(ctx, dstRect, cgImage);
      resized = CGBitmapContextCreateImage(ctx);
      CGContextRelease(ctx);
      didResize = YES;
    }
  }

  // Extract pixel data
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  NSMutableData *pixelData = [NSMutableData dataWithLength:modelW * modelH * 4];
  CGContextRef bitmapCtx = CGBitmapContextCreate(
    pixelData.mutableBytes, modelW, modelH, 8, modelW * 4,
    colorSpace, kCGImageAlphaPremultipliedLast);
  CGColorSpaceRelease(colorSpace);

  if (!bitmapCtx) {
    if (didResize) CGImageRelease(resized);
    return nil;
  }

  CGRect drawRect = CGRectMake(0, 0, modelW, modelH);
  CGContextDrawImage(bitmapCtx, drawRect, resized);
  CGContextRelease(bitmapCtx);

  if (didResize) CGImageRelease(resized);

  // Create NCHW Float32 MLMultiArray
  NSError *err = nil;
  MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                dataType:MLMultiArrayDataTypeFloat32
                                                   error:&err];
  if (!arr || err) return nil;

  uint8_t *pixels = (uint8_t *)pixelData.bytes;
  float *outPtr = (float *)arr.dataPointer;
  int plane = modelW * modelH;

  for (int y = 0; y < modelH; y++) {
    for (int x = 0; x < modelW; x++) {
      int srcIdx = (y * modelW + x) * 4;
      int dstIdx = y * modelW + x;
      float b = (pixels[srcIdx + 0] - 127.5f) / 127.5f;
      float g = (pixels[srcIdx + 1] - 127.5f) / 127.5f;
      float r = (pixels[srcIdx + 2] - 127.5f) / 127.5f;
      outPtr[0 * plane + dstIdx] = r;
      outPtr[1 * plane + dstIdx] = g;
      outPtr[2 * plane + dstIdx] = b;
    }
  }

  return [MLFeatureValue featureValueWithMultiArray:arr];
}

// ─── Private: EXIF Orientation Fix ─────────────────────────────────────────────

- (UIImage *)fixOrientation:(UIImage *)image {
  if (image.imageOrientation == UIImageOrientationUp) return image;

  UIGraphicsBeginImageContextWithOptions(image.size, NO, image.scale);
  [image drawInRect:CGRectMake(0, 0, image.size.width, image.size.height)];
  UIImage *normalized = UIGraphicsGetImageFromCurrentImageContext();
  UIGraphicsEndImageContext();
  return normalized ?: image;
}

@end
