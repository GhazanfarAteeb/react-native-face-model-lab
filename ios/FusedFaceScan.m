//
//  FusedFaceScan.m — Obj-C bridge for the NativeFusedFaceScan TurboModule.
//
#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(NativeFusedFaceScan, NSObject)

RCT_EXTERN_METHOD(loadModel:(NSString *)assetName
                  inputName:(NSString *)inputName
                  outputName:(NSString *)outputName
                  inputWidth:(double)inputWidth
                  inputHeight:(double)inputHeight
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(scanPhoto:(NSString *)imagePath
                  paddingRatio:(double)paddingRatio
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(release:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(getDeviceInfo:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

@end
