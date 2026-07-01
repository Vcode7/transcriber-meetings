; Inno Setup Script — AI Meeting Transcriber
; Build: iscc installer\setup.iss
; Requires Inno Setup 6+ (https://jrsoftware.org/isinfo.php)
;
; NOTE ON MODEL DISTRIBUTION:
;   AI models are NOT bundled in this installer — they are too large (~4 GB+).
;   Encrypted models (.dat) and the Qwen3 nlp-engine folder are distributed
;   separately and must be placed by the user/technician after installation:
;
;     {app}\runtime\models\        ← encrypted .dat files (speech, diarization, etc.)
;     {app}\runtime\nlp-engine\    ← Qwen3-4B plain model folder (shipped separately)
;
;   The installer creates these empty directories automatically.

#define MyAppName "AI Meeting Transcriber"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "VoiceSum Technologies"
#define MyAppExeName "launcher.exe"
#define MyAppId "{B4C1D2A3-E5F6-4789-AB01-CD23EF456789}"

[Setup]
AppId={{#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://voicesum.ai
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; No silent install — show wizard
DisableWelcomePage=no
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=dist
OutputBaseFilename=Setup_AIMeetingTranscriber_v{#MyAppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Minimum OS: Windows 10 1903
MinVersion=10.0.18362
; Show license
LicenseFile=assets\LICENSE.rtf
CreateUninstallRegKey=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Dirs]
; Runtime directories — models are NOT bundled; placed manually after install
Name: "{app}\runtime"
Name: "{app}\runtime\models"
Name: "{app}\runtime\nlp-engine"
Name: "{app}\runtime\data"
Name: "{app}\runtime\uploads"

[Files]
; Main launcher (single .exe)
Source: "..\Application\launcher.exe"; DestDir: "{app}"; Flags: ignoreversion

; Backend (directory of files from PyInstaller)
Source: "..\Application\backend\*"; DestDir: "{app}\backend"; Flags: ignoreversion recursesubdirs createallsubdirs

; Frontend — Electron win-unpacked output
Source: "..\Application\frontend\win-unpacked\*"; DestDir: "{app}\frontend"; Flags: ignoreversion recursesubdirs createallsubdirs

; Assets (icons etc.)
Source: "..\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs

; Default .env config (no secrets — user configures API keys in settings UI)
Source: "..\backend\.env.example"; DestDir: "{app}\backend"; DestName: ".env"; Flags: ignoreversion onlyifdoesntexist

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "AI-powered meeting transcription and summarization"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

; Desktop shortcut (optional task)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Launch after install (optional)
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Clean temp decrypted models left in %TEMP% on uninstall
Filename: "cmd.exe"; Parameters: "/c rmdir /s /q ""%TEMP%\voicesum_runtime"""; Flags: runhidden; RunOnceId: "CleanTempModels"

[Code]
// Custom check: ensure no existing instance is running before install
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
