//
//  VisionFaceDetect.mm — Apple Vision face detector for FaceModelLab (iOS).
//
//  Uses VNDetectFaceRectanglesRequest (fast, no landmarks) or
//  VNDetectFaceLandmarksRequest (slower, 5 landmarks for ArcFace alignment).
//  Exposed to JS as "NativeVisionFaceDetect" with one async method: detect.
//
//  This is the native Apple Vision alternative to the JS-based ONNX detectors (YuNet,
//  SCRFD, BlazeFace). It runs on the Neural Engine via the Vision framework and provides
//  bounding boxes + optional landmarks in the original image coordinate space.
//
#import <React/RCTBridgeModule.h>
#import <Vision/Vision.h>
#import <Foundation/Foundation.h>

@interface NativeVisionFaceDetect : NSObject <RCTBridgeModule>
@end

@implementation NativeVisionFaceDetect

RCT_EXPORT_MODULE(NativeVisionFaceDetect);

+ (BOOL)requiresMainQueueSetup { return NO; }

- (dispatch_queue_t)methodQueue {
  return dispatch_queue_create("com.facemodellab.visiondetect", DISPATCH_QUEUE_SERIAL);
}

RCT_EXPORT_METHOD(detect:(NSString *)imagePath
                  minFaceSize:(double)minFaceSize
                  useLandmarks:(BOOL)useLandmarks
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  @try {
    NSURL *url = [NSURL fileURLWithPath:imagePath];
    NSData *data = [NSData dataWithContentsOfURL:url];
    if (!data) {
      reject(@"file_error", @"Could not read image file", nil);
      return;
    }

    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
    if (!source) {
      reject(@"decode_error", @"Could not decode image", nil);
      return;
    }
    CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (!cgImage) {
      reject(@"decode_error", @"Could not create CGImage", nil);
      return;
    }

    size_t imgW = CGImageGetWidth(cgImage);
    size_t imgH = CGImageGetHeight(cgImage);
    CGSize imgSize = CGSizeMake(imgW, imgH);

    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block NSArray *results = @[];
    __block NSError *blockErr = nil;

    if (useLandmarks) {
      // VNDetectFaceLandmarksRequest — 5 landmarks for ArcFace alignment.
      VNDetectFaceLandmarksRequest *req = [[VNDetectFaceLandmarksRequest alloc]
        initWithCompletionHandler:^(VNRequest *request, NSError *error) {
          if (error) {
            blockErr = error;
            dispatch_semaphore_signal(sem);
            return;
          }
          NSMutableArray *faces = [NSMutableArray array];
          for (VNFaceObservation *obs in request.results) {
            CGRect bbox = obs.boundingBox;
            // Convert from Vision coords (origin bottom-left, normalized) to image coords
            // (origin top-left, pixels). Vision normalizes to [0,1] with Y-up.
            double left = bbox.origin.x * imgW;
            double top = (1.0 - bbox.origin.y - bbox.size.height) * imgH;
            double w = bbox.size.width * imgW;
            double h = bbox.size.height * imgH;

            // Filter by min face size
            if (MIN(w, h) < minFaceSize) continue;

            NSMutableDictionary *face = [@{
              @"left": @(left),
              @"top": @(top),
              @"width": @(w),
              @"height": @(h),
              @"yaw": @((obs.yaw ?: @0).doubleValue),
              @"pitch": @((obs.pitch ?: @0).doubleValue),
              @"roll": @((obs.roll ?: @0).doubleValue),
            } mutableCopy];

            VNFaceLandmarks2D *landmarks = (VNFaceLandmarks2D *)obs.landmarks;
            if (landmarks) {
              // Helper to extract the first point from a landmark region.
              // pointsInImage returns a C array of CGPoint in image pixel coords (Y-down).
              auto addLandmark = ^(NSString *key, VNFaceLandmarkRegion2D *region) {
                if (region && region.pointCount > 0) {
                  const CGPoint *pts = [region pointsInImageOfSize:imgSize];
                  if (pts) {
                    face[key] = @{
                      @"x": @(pts[0].x),
                      @"y": @(pts[0].y),
                    };
                  }
                }
              };
              addLandmark(@"leftEye", landmarks.leftEye);
              addLandmark(@"rightEye", landmarks.rightEye);
              addLandmark(@"noseBase", landmarks.nose);
              addLandmark(@"mouthLeft", landmarks.innerLips);
              addLandmark(@"mouthRight", landmarks.outerLips);
            }

            [faces addObject:face];
          }
          results = [faces copy];
          dispatch_semaphore_signal(sem);
        }];

      VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
        initWithCGImage:cgImage options:@{}];
      NSError *err = nil;
      [handler performRequests:@[req] error:&err];
      if (err) {
        CGImageRelease(cgImage);
        reject(@"vision_error", err.localizedDescription, err);
        return;
      }
    } else {
      // VNDetectFaceRectanglesRequest — fast bounding boxes only (no landmarks).
      VNDetectFaceRectanglesRequest *req = [[VNDetectFaceRectanglesRequest alloc]
        initWithCompletionHandler:^(VNRequest *request, NSError *error) {
          if (error) {
            blockErr = error;
            dispatch_semaphore_signal(sem);
            return;
          }
          NSMutableArray *faces = [NSMutableArray array];
          for (VNFaceObservation *obs in request.results) {
            CGRect bbox = obs.boundingBox;
            double left = bbox.origin.x * imgW;
            double top = (1.0 - bbox.origin.y - bbox.size.height) * imgH;
            double w = bbox.size.width * imgW;
            double h = bbox.size.height * imgH;

            if (MIN(w, h) < minFaceSize) continue;

            [faces addObject:@{
              @"left": @(left),
              @"top": @(top),
              @"width": @(w),
              @"height": @(h),
              @"yaw": @((obs.yaw ?: @0).doubleValue),
              @"pitch": @((obs.pitch ?: @0).doubleValue),
              @"roll": @((obs.roll ?: @0).doubleValue),
            }];
          }
          results = [faces copy];
          dispatch_semaphore_signal(sem);
        }];

      VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
        initWithCGImage:cgImage options:@{}];
      NSError *err = nil;
      [handler performRequests:@[req] error:&err];
      if (err) {
        CGImageRelease(cgImage);
        reject(@"vision_error", err.localizedDescription, err);
        return;
      }
    }

    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    CGImageRelease(cgImage);

    if (blockErr) {
      reject(@"vision_error", blockErr.localizedDescription, blockErr);
      return;
    }

    resolve(@{
      @"faces": results,
      @"imageWidth": @(imgW),
      @"imageHeight": @(imgH),
    });
  } @catch (NSException *e) {
    reject(@"exception", e.reason, nil);
  }
}

@end
