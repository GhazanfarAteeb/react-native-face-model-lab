package com.facemodellab

/**
 * ScanForegroundService — keeps the gallery scan running when the app is backgrounded.
 *
 * On Android the OS suspends a backgrounded process within seconds, which freezes the JS scan
 * loop mid-photo. A foreground service (with its persistent notification) plus a PARTIAL wake
 * lock is the sanctioned way to keep the process + CPU alive so the scan — and the selected
 * CPU/NNAPI execution provider — keeps making progress with the screen off or the app switched
 * away. The scan itself still runs in JS/native as before; this class only prevents the OS from
 * killing it. Lifecycle is driven from JS (BackgroundScanModule): start() when a scan begins,
 * updateNotification() on progress, stop() when it finishes/cancels.
 */
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat

class ScanForegroundService : Service() {
  private var wakeLock: PowerManager.WakeLock? = null
  private var title: String = "Scanning photos"

  companion object {
    const val CHANNEL_ID = "fml_scan"
    const val NOTIF_ID = 4711
    const val ACTION_START = "com.facemodellab.scan.START"
    const val ACTION_STOP = "com.facemodellab.scan.STOP"
    const val EXTRA_TITLE = "title"
    const val EXTRA_TEXT = "text"

    /** Live instance so the JS module can push notification updates / stop without re-delivering
     *  intents. Null whenever the service isn't running. */
    @Volatile
    var instance: ScanForegroundService? = null
  }

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onCreate() {
    super.onCreate()
    instance = this
  }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    if (intent?.action == ACTION_STOP) {
      stopScan()
      return START_NOT_STICKY
    }
    title = intent?.getStringExtra(EXTRA_TITLE) ?: title
    val text = intent?.getStringExtra(EXTRA_TEXT) ?: "Scanning your photos…"
    startInForeground(text)
    acquireWakeLock()
    // Don't auto-restart with a stale intent if the OS kills us under memory pressure — the JS
    // side owns scan state and resumes from its on-disk checkpoint on the next launch.
    return START_NOT_STICKY
  }

  private fun ensureChannel() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      val mgr = getSystemService(NotificationManager::class.java)
      if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
        val ch = NotificationChannel(CHANNEL_ID, "Photo scan", NotificationManager.IMPORTANCE_LOW)
        ch.setShowBadge(false)
        mgr.createNotificationChannel(ch)
      }
    }
  }

  private fun buildNotification(text: String): Notification {
    ensureChannel()
    val launch = packageManager.getLaunchIntentForPackage(packageName)
    val contentIntent = launch?.let {
      PendingIntent.getActivity(
        this, 0, it,
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
      )
    }
    return NotificationCompat.Builder(this, CHANNEL_ID)
      .setContentTitle(title)
      .setContentText(text)
      .setSmallIcon(android.R.drawable.stat_notify_sync)
      .setOngoing(true)
      .setOnlyAlertOnce(true)
      .setPriority(NotificationCompat.PRIORITY_LOW)
      .setContentIntent(contentIntent)
      .build()
  }

  private fun startInForeground(text: String) {
    val notification = buildNotification(text)
    val type =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
      else 0
    ServiceCompat.startForeground(this, NOTIF_ID, notification, type)
  }

  /** Refresh the notification text (called from the JS module on scan progress). */
  fun updateNotification(text: String) {
    val mgr = getSystemService(NotificationManager::class.java) ?: return
    mgr.notify(NOTIF_ID, buildNotification(text))
  }

  private fun acquireWakeLock() {
    if (wakeLock?.isHeld == true) return
    val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
    wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "FaceModelLab:scan").apply {
      setReferenceCounted(false)
      // 60-min safety cap so a missed stop() can't pin the CPU forever; released explicitly on stop.
      acquire(60 * 60 * 1000L)
    }
  }

  private fun releaseWakeLock() {
    try {
      wakeLock?.let { if (it.isHeld) it.release() }
    } catch (_: Exception) {
      // ignore — already released or never acquired
    }
    wakeLock = null
  }

  fun stopScan() {
    releaseWakeLock()
    ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
    stopSelf()
  }

  override fun onDestroy() {
    releaseWakeLock()
    instance = null
    super.onDestroy()
  }
}
