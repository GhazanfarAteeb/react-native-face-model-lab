package com.facemodellab

/**
 * BackgroundScanModule — thin JS bridge for the scan foreground service.
 *
 * JS (src/runtime/BackgroundScan.ts) calls start() when a scan begins, updateProgress() as
 * photos complete, and stop() when it finishes or is cancelled. start() must be called while the
 * app is foreground (Android 12+ forbids starting a foreground service from the background); the
 * scan is always kicked off from a user tap, so that holds.
 */
import android.content.Intent
import android.os.Build
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

class BackgroundScanModule(private val reactCtx: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactCtx) {

  override fun getName() = "BackgroundScan"

  @ReactMethod
  fun start(title: String, text: String, promise: Promise) {
    try {
      val intent = Intent(reactCtx, ScanForegroundService::class.java).apply {
        action = ScanForegroundService.ACTION_START
        putExtra(ScanForegroundService.EXTRA_TITLE, title)
        putExtra(ScanForegroundService.EXTRA_TEXT, text)
      }
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        reactCtx.startForegroundService(intent)
      } else {
        reactCtx.startService(intent)
      }
      promise.resolve(true)
    } catch (e: Exception) {
      promise.reject("start_failed", e)
    }
  }

  @ReactMethod
  fun updateProgress(text: String) {
    ScanForegroundService.instance?.updateNotification(text)
  }

  @ReactMethod
  fun stop() {
    val svc = ScanForegroundService.instance
    if (svc != null) {
      svc.stopScan()
    } else {
      // Instance ref lost (rare) — deliver a stop intent as a fallback.
      try {
        reactCtx.startService(
          Intent(reactCtx, ScanForegroundService::class.java).apply {
            action = ScanForegroundService.ACTION_STOP
          },
        )
      } catch (_: Exception) {
        // nothing running to stop
      }
    }
  }
}
