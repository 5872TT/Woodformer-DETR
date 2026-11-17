param(
    [Parameter(Mandatory=$true)][string]$Data,
    [ValidateSet('baseline','fewd')][string]$Model = 'baseline',
    [int]$Epochs = 80, [int]$BatchSize = 4, [int]$Workers = 0,
    [string]$Python = 'python', [string]$Resume = '', [string]$Pretrained = '',
    [switch]$Check, [switch]$Amp
)
$ErrorActionPreference = 'Stop'
$trainScript = Join-Path $PSScriptRoot 'RT-DETR-main\rtdetr_pytorch\tools\train_oulu.py'
$trainArgs = @($trainScript,'--data',$Data,'--model',$Model,'--epochs',$Epochs,'--batch-size',$BatchSize,'--workers',$Workers)
if ($Resume) { $trainArgs += @('--resume',$Resume) }
if ($Pretrained) { $trainArgs += @('--pretrained',$Pretrained) }
if ($Check) { $trainArgs += '--check' }
if ($Amp) { $trainArgs += '--amp' }
& $Python @trainArgs
exit $LASTEXITCODE
