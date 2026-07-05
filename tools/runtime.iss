; ============================================================
; VoiceSum Runtime Installer
; Installs embeddable Python 3.12 + ML deps + ffmpeg to
; %ProgramData%\VoiceSum\runtime\
;
; This installer is built ONCE when ML dependencies change.
; It is distributed separately from the main application installer.
;
; Build:
;   iscc tools\runtime.iss
;
; Output:
;   tools\runtime_dist\VoiceSum-Runtime-2.0.exe
; ============================================================

#define RuntimeVersion "2.0"
#define PyVersion      "3.12.3"
#define AppPublisher   "VoiceSum Technologies"
#define RuntimeId      "{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"

[Setup]
AppId={{#RuntimeId}
AppName=VoiceSum Runtime {#RuntimeVersion}
AppVersion={#RuntimeVersion}
AppPublisher={#AppPublisher}
; Install into %ProgramData%\VoiceSum\runtime\
DefaultDirName={commonappdata}\VoiceSum\runtime
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableWelcomePage=no
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=runtime_dist
OutputBaseFilename=VoiceSum-Runtime-{#RuntimeVersion}
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.18362
CreateUninstallRegKey=yes
UninstallDisplayName=VoiceSum Runtime {#RuntimeVersion}

; Registry key to mark runtime as installed (read by app launcher)
; HKLM\SOFTWARE\VoiceSum\Runtime  ->  Version = "2.0"
;                                     InstallPath = "C:\ProgramData\VoiceSum\runtime"

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Dirs]
Name: "{app}\python"
Name: "{app}\python\Lib"
Name: "{app}\python\Lib\site-packages"

[Files]
; Embeddable Python + all pip-installed ML deps
; These are placed in build\runtime-pkg\ by build_runtime.bat
Source: "..\build\runtime-pkg\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

; ffmpeg / ffprobe
Source: "..\build\runtime-pkg\ffmpeg.exe";  DestDir: "{app}"; Flags: ignoreversion
Source: "..\build\runtime-pkg\ffprobe.exe"; DestDir: "{app}"; Flags: ignoreversion

; Version stamp (read by launcher.exe to check compatibility)
Source: "..\build\runtime-pkg\runtime-version.txt"; DestDir: "{app}"; Flags: ignoreversion

[Registry]
; Write runtime info to registry for easy discovery by the app
Root: HKLM; Subkey: "SOFTWARE\VoiceSum\Runtime"; ValueType: string; ValueName: "Version";     ValueData: "{#RuntimeVersion}"; Flags: createvalueifdoesntexist uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\VoiceSum\Runtime"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: createvalueifdoesntexist

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
// Check if a newer runtime is already installed ? warn before downgrading
function InitializeSetup(): Boolean;
var
  ExistingVer: String;
begin
  Result := True;
  if RegQueryStringValue(HKLM, 'SOFTWARE\VoiceSum\Runtime', 'Version', ExistingVer) then
  begin
    if CompareStr(ExistingVer, '{#RuntimeVersion}') > 0 then
    begin
      if MsgBox(
        'A newer VoiceSum Runtime (' + ExistingVer + ') is already installed.' + #13#10 +
        'Installing version {#RuntimeVersion} may downgrade it.' + #13#10#13#10 +
        'Continue anyway?',
        mbConfirmation, MB_YESNO) = IDNO then
        Result := False;
    end;
  end;
end;
