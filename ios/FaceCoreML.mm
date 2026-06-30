//
//  FaceCoreML.mm — native CoreML embedder for FaceModelLab (iOS).
//
//  Runs the face-embedding model on the Apple Neural Engine via CoreML, in parallel with the
//  ORT-CPU embedder (see src/runtime/HybridEmbedder.ts). Exposed to JS as "NativeFaceCoreML"
//  with three async methods: load / infer / release.
//
//  Models are bundled as .mlpackage and compiled by Xcode to <name>.mlmodelc at build time;
//  we load that compiled model with computeUnits = All so CoreML can schedule the ANE.
//
//  Registered as a legacy RCTBridgeModule — under the New Architecture this is surfaced through
//  the TurboModule interop layer automatically, and the JS side (NativeFaceCoreML.ts) resolves
//  it from either registry.
//
#import <React/RCTBridgeModule.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

@interface NativeFaceCoreML : NSObject <RCTBridgeModule>
@end

@implementation NativeFaceCoreML {
  NSMutableDictionary<NSNumber *, MLModel *> *_models;
  NSMutableDictionary<NSNumber *, NSString *> *_inNames;  // resolved input feature name per handle
  NSMutableDictionary<NSNumber *, NSString *> *_outNames; // resolved output feature name per handle
  NSInteger _nextHandle;
}

RCT_EXPORT_MODULE(NativeFaceCoreML);

- (instancetype)init {
  if (self = [super init]) {
    _models = [NSMutableDictionary new];
    _inNames = [NSMutableDictionary new];
    _outNames = [NSMutableDictionary new];
    _nextHandle = 0;
  }
  return self;
}

+ (BOOL)requiresMainQueueSetup { return NO; }

// Serial queue dedicated to this module — CoreML work runs off the JS thread, and (being a
// different queue from ORT's) overlaps with ORT-CPU inference. That overlap is the whole point.
- (dispatch_queue_t)methodQueue {
  return dispatch_queue_create("com.facemodellab.coreml", DISPATCH_QUEUE_SERIAL);
}

RCT_EXPORT_METHOD(load:(NSString *)assetName
                  inputName:(NSString *)inputName
                  outputName:(NSString *)outputName
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  @try {
    NSString *base = [assetName stringByDeletingPathExtension];
    NSURL *url = [[NSBundle mainBundle] URLForResource:base withExtension:@"mlmodelc"];
    if (!url) {
      resolve(@(-1)); // model not bundled (or not compiled) — JS falls back to CPU
      return;
    }
    MLModelConfiguration *cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsAll; // lets CoreML schedule the Neural Engine
    NSError *err = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!model || err) {
      resolve(@(-1));
      return;
    }

    // Resolve the actual feature names now (honour the names JS passed, else use the first).
    NSDictionary<NSString *, MLFeatureDescription *> *ins = model.modelDescription.inputDescriptionsByName;
    NSDictionary<NSString *, MLFeatureDescription *> *outs = model.modelDescription.outputDescriptionsByName;
    NSString *inName = (inputName.length > 0 && ins[inputName]) ? inputName : ins.allKeys.firstObject;
    NSString *outName = (outputName.length > 0 && outs[outputName]) ? outputName : outs.allKeys.firstObject;
    if (!inName || !outName) {
      resolve(@(-1));
      return;
    }

    NSInteger h = _nextHandle++;
    _models[@(h)] = model;
    _inNames[@(h)] = inName;
    _outNames[@(h)] = outName;
    resolve(@(h));
  } @catch (NSException *e) {
    resolve(@(-1));
  }
}

RCT_EXPORT_METHOD(infer:(double)handle
                  input:(NSArray<NSNumber *> *)input
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  NSNumber *key = @((NSInteger)handle);
  MLModel *model = _models[key];
  NSString *inName = _inNames[key];
  NSString *outName = _outNames[key];
  if (!model || !inName || !outName) {
    reject(@"no_model", @"invalid CoreML handle", nil);
    return;
  }
  NSError *err = nil;

  MLFeatureDescription *inDesc = model.modelDescription.inputDescriptionsByName[inName];
  NSArray<NSNumber *> *shape = inDesc.multiArrayConstraint.shape;
  if (!shape) {
    reject(@"bad_input", @"model input is not a multiarray", nil);
    return;
  }

  MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                 dataType:MLMultiArrayDataTypeFloat32
                                                    error:&err];
  if (!arr || err) {
    reject(@"alloc", err.localizedDescription ?: @"MLMultiArray alloc failed", err);
    return;
  }
  // Our preprocessed tensor is already flattened in the model's layout, so a contiguous copy
  // into the (C-contiguous) MLMultiArray is correct. Guard against any length mismatch.
  float *ptr = (float *)arr.dataPointer;
  NSUInteger n = MIN((NSUInteger)arr.count, input.count);
  for (NSUInteger i = 0; i < n; i++) {
    ptr[i] = input[i].floatValue;
  }

  MLDictionaryFeatureProvider *fp =
      [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        inName : [MLFeatureValue featureValueWithMultiArray:arr]
      }
                                                        error:&err];
  if (!fp || err) {
    reject(@"provider", err.localizedDescription ?: @"feature provider failed", err);
    return;
  }

  id<MLFeatureProvider> out = [model predictionFromFeatures:fp error:&err];
  if (!out || err) {
    reject(@"infer", err.localizedDescription ?: @"prediction failed", err);
    return;
  }

  MLMultiArray *res = [out featureValueForName:outName].multiArrayValue;
  if (!res) {
    reject(@"no_output", @"model output is not a multiarray", nil);
    return;
  }

  NSUInteger count = (NSUInteger)res.count;
  NSMutableArray<NSNumber *> *vec = [NSMutableArray arrayWithCapacity:count];
  for (NSUInteger i = 0; i < count; i++) {
    [vec addObject:@([res[i] floatValue])]; // flat indexed subscript; handles fp16/fp32 outputs
  }
  resolve(vec);
}

RCT_EXPORT_METHOD(release:(double)handle
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  NSNumber *key = @((NSInteger)handle);
  [_models removeObjectForKey:key];
  [_inNames removeObjectForKey:key];
  [_outNames removeObjectForKey:key];
  resolve(nil);
}

@end
