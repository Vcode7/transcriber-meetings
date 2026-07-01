/**
 * Electron main process — AI Meeting Transcriber frontend wrapper
 *
 * Wraps the Vite/React frontend in an Electron window.
 * The backend must already be running (started by launcher.exe).
 *
 * Build:
 *   cd frontend-electron && npm install && npm run dist
 */
const { app, BrowserWindow, shell, Menu, session, desktopCapturer } = require('electron')
const path = require('path')
const fs = require('fs')

// Log path: Application/frontend.log (when packaged) or local folder (dev)
let logPath = path.join(__dirname, 'frontend.log')
if (app.isPackaged) {
  logPath = path.join(path.dirname(app.getPath('exe')), '..', '..', 'frontend.log')
}

function writeLog(message) {
  const timestamp = new Date().toISOString()
  const formatted = `[${timestamp}] ${message}\n`
  try {
    fs.appendFileSync(logPath, formatted, 'utf8')
  } catch (e) {
    console.error('Failed to write log:', e)
  }
}

// Redirect console logs to file
const originalLog = console.log
const originalError = console.error
const originalWarn = console.warn

console.log = (...args) => {
  writeLog(`[Main] [INFO] ${args.join(' ')}`)
  originalLog(...args)
}
console.error = (...args) => {
  writeLog(`[Main] [ERROR] ${args.join(' ')}`)
  originalError(...args)
}
console.warn = (...args) => {
  writeLog(`[Main] [WARN] ${args.join(' ')}`)
  originalWarn(...args)
}

process.on('uncaughtException', (error) => {
  writeLog(`[Main] [CRASH] Uncaught Exception: ${error.stack || error}`)
})


// Backend URL — always localhost
const BACKEND_URL = 'http://127.0.0.1:8000'
// Frontend — served from backend or from local file
const FRONTEND_URL = process.env.FRONTEND_DEV_URL || `${BACKEND_URL}/app`

// For the demo, we serve the built React app from the backend static files mount,
// OR we can serve the index.html directly from the packaged dist/ folder.
// Adjust SERVE_LOCAL to true to serve from local files.
const SERVE_LOCAL = true
const LOCAL_DIST = path.join(__dirname, 'dist', 'index.html')

let mainWindow = null

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 600,
    title: 'AI Meeting Transcriber',
    backgroundColor: '#0b0d17',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
    },
    // Icon
    // icon: path.join(__dirname, '..', 'assets', 'icon.ico'),
    show: false,
    autoHideMenuBar: true,
  })

  // Remove menu bar (production)
  Menu.setApplicationMenu(null)

  if (SERVE_LOCAL) {
    mainWindow.loadFile(LOCAL_DIST)
  } else {
    mainWindow.loadURL(FRONTEND_URL)
  }

  // Show when ready to prevent white flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
    mainWindow.focus()
  })

  // Log any file load errors
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    console.error('Page failed to load:', errorCode, errorDescription, validatedURL)
  })


  // Open external links in browser (not Electron)
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http')) {
      shell.openExternal(url)
      return { action: 'deny' }
    }
    return { action: 'allow' }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

app.whenReady().then(() => {
  // Set up screen/tab capture handler for getDisplayMedia()
  session.defaultSession.setDisplayMediaRequestHandler(async (request, callback) => {
    try {
      const sources = await desktopCapturer.getSources({ types: ['screen', 'window'] })
      if (sources.length > 0) {
        callback({
          video: sources[0],
          audio: 'loopback' // Captures system loopback audio on Windows!
        })
      } else {
        callback({ error: 'No display capture sources found.' })
      }
    } catch (err) {
      console.error('Error in display media handler:', err)
      callback({ error: err.message })
    }
  })

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  app.quit()
})

// Security: prevent navigation to external URLs
app.on('web-contents-created', (_, contents) => {
  contents.on('will-navigate', (event, url) => {
    if (!url.startsWith('http://127.0.0.1') && !url.startsWith('file://')) {
      event.preventDefault()
    }
  })
})
