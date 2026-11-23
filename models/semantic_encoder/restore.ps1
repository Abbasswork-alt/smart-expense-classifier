$ErrorActionPreference = "Stop"
$repo = "https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"
$assets = @{
    "onnx/model_quantized.onnx" = "66FC00F5F29AFCAFF34092E1BDD20008CA3918265A82FB9695A551E510CC4EBC"
    "tokenizer.json" = "B60B6B43406A48BF3638526314F3D232D97058BC93472FF2DE930D43686FA441"
}
foreach ($name in $assets.Keys) {
    $destination = Join-Path $PSScriptRoot ([IO.Path]::GetFileName($name))
    if ((Test-Path -LiteralPath $destination) -and
        (Get-FileHash -LiteralPath $destination).Hash -eq $assets[$name]) { continue }
    $temp = "$destination.$([guid]::NewGuid()).tmp"
    try {
        Invoke-WebRequest -Uri "$repo/$name" -OutFile $temp
        if ((Get-FileHash -LiteralPath $temp).Hash -ne $assets[$name]) { throw "Unexpected hash for $name" }
        Move-Item -LiteralPath $temp -Destination $destination -Force
    } finally {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
    }
}
