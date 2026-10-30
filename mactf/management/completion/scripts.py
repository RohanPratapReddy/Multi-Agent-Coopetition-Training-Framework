"""Completion scripts. ASCII only, ``\\n`` line endings (``write_raw`` keeps them on Windows).

Document 16, section 9.4. Each script reconstructs the words up to the cursor, calls
``<typed program> __complete <shell> -- <words>``, splits the directive from the candidates,
and hands them to the shell. ``@PROG@`` and ``@FUNC@`` are the only substitutions.
"""

from __future__ import annotations

import re

from ...exceptions import CommandError
from . import SHELLS

_BASH = r"""# bash completion for @PROG@
#   current shell:  eval "$(@PROG@ completion bash)"
#   every shell:    @PROG@ completion bash >> ~/.bashrc
@FUNC@() {
    local line cur out directive body item trim ch head i
    local -a words=()
    line="${COMP_LINE:0:COMP_POINT}"
    read -r -a words <<< "$line"
    if [[ "$line" == *[[:space:]] ]]; then
        words+=("")
    fi
    cur="${words[${#words[@]}-1]}"
    out="$("${words[0]}" __complete bash -- "${words[@]}" 2>/dev/null)" || return 0
    directive="${out%%$'\n'*}"
    body="${out#*$'\n'}"
    [[ "$body" == "$out" ]] && body=""
    directive="${directive#:}"
    [[ "$directive" =~ ^[0-9]+$ ]] || directive=0
    COMPREPLY=()
    while IFS= read -r item; do
        item="${item%$'\r'}"
        [[ -n "$item" ]] && COMPREPLY+=("${item%%$'\t'*}")
    done <<< "$body"
    trim=""
    for ch in = :; do
        if [[ "$COMP_WORDBREAKS" == *"$ch"* && "$cur" == *"$ch"* ]]; then
            head="${cur%"$ch"*}$ch"
            (( ${#head} > ${#trim} )) && trim="$head"
        fi
    done
    if [[ -n "$trim" ]]; then
        for i in "${!COMPREPLY[@]}"; do
            COMPREPLY[i]="${COMPREPLY[i]#"$trim"}"
        done
    fi
    (( directive & 2 )) && compopt -o nospace 2>/dev/null
    if (( ${#COMPREPLY[@]} == 0 )) && (( directive & 5 )); then
        compopt -o filenames 2>/dev/null
        if (( directive & 4 )); then
            while IFS= read -r item; do COMPREPLY+=("$item"); done < <(compgen -d -- "$cur")
        else
            while IFS= read -r item; do COMPREPLY+=("$item"); done < <(compgen -f -- "$cur")
        fi
    fi
    return 0
}
complete -F @FUNC@ @PROG@
"""

_ZSH = r"""#compdef @PROG@
# zsh completion for @PROG@ (run compinit first)
#   current shell:  eval "$(@PROG@ completion zsh)"
#   every shell:    @PROG@ completion zsh > "${fpath[1]}/@FUNC@"
@FUNC@() {
    local out directive item value description
    local -a lines candidates flags
    out="$("${words[1]}" __complete zsh -- "${(@)words[1,CURRENT]}" 2>/dev/null)" || return 0
    lines=("${(@f)out}")
    directive="${lines[1]#:}"
    [[ "$directive" == <-> ]] || directive=0
    for item in "${(@)lines[2,-1]}"; do
        [[ -n "$item" ]] || continue
        value="${item%%$'\t'*}"
        description=""
        [[ "$item" == *$'\t'* ]] && description="${item#*$'\t'}"
        value="${value//:/\\:}"
        if [[ -n "$description" ]]; then
            candidates+=("${value}:${description}")
        else
            candidates+=("$value")
        fi
    done
    (( directive & 2 )) && flags=(-S '')
    if (( ${#candidates} > 0 )); then
        _describe -t values '@PROG@' candidates "${flags[@]}"
    elif (( directive & 4 )); then
        _files -/
    elif (( directive & 1 )); then
        _files
    fi
}
if [ "$funcstack[1]" = "@FUNC@" ]; then
    @FUNC@ "$@"
else
    compdef @FUNC@ @PROG@
fi
"""

_FISH = r"""# fish completion for @PROG@
#   install:  @PROG@ completion fish > ~/.config/fish/completions/@PROG@.fish
function @FUNC@
    set -l tokens (commandline -opc)
    set -l current (commandline -ct)
    set -l out (command $tokens[1] __complete fish -- $tokens $current 2>/dev/null)
    test (count $out) -ge 1; or return 0
    set -l directive (string replace -r '^:' '' -- $out[1])
    set -e out[1]
    if test (count $out) -gt 0
        printf '%s\n' $out
        return 0
    end
    if contains -- $directive 4 5 6 7
        __fish_complete_directories $current
    else if contains -- $directive 1 3
        __fish_complete_path $current
    end
end
complete -c @PROG@ -f -a '(@FUNC@)'
"""

_POWERSHELL = r"""# PowerShell completion for @PROG@
#   current session:  @PROG@ completion powershell | Out-String | Invoke-Expression
#   every session:    add the line above to your $PROFILE
Register-ArgumentCompleter -Native -CommandName '@PROG@' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $words = New-Object System.Collections.Generic.List[string]
    $lastEnd = -1
    foreach ($element in $commandAst.CommandElements) {
        if ($element.Extent.StartOffset -ge $cursorPosition) { break }
        $text = $element.Extent.Text
        if ($text.Length -ge 2 -and ($text[0] -eq "'" -or $text[0] -eq '"') -and $text[$text.Length - 1] -eq $text[0]) {
            $text = $text.Substring(1, $text.Length - 2)
        }
        $words.Add($text)
        $lastEnd = $element.Extent.EndOffset
    }
    $newWord = $true
    if ($words.Count -gt 0 -and $cursorPosition -le $lastEnd) {
        $words[$words.Count - 1] = $wordToComplete
        $newWord = $false
    }
    $program = $words[0]
    $arguments = @('__complete', 'powershell')
    if ($newWord) { $arguments += '--new-word' }
    $arguments += '--'
    $arguments += $words.ToArray()
    $lines = @(& $program @arguments 2>$null)
    foreach ($line in ($lines | Select-Object -Skip 1)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line.Split([char]9, 2)
        $value = $parts[0]
        $tip = if ($parts.Length -gt 1 -and $parts[1]) { $parts[1] } else { $value }
        $insert = if ($value -match '\s') { "'" + $value.Replace("'", "''") + "'" } else { $value }
        [System.Management.Automation.CompletionResult]::new($insert, $value, 'ParameterValue', $tip)
    }
}
"""

SCRIPTS = {"bash": _BASH, "zsh": _ZSH, "fish": _FISH, "powershell": _POWERSHELL}


def function_name(shell: str, prog: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]", "_", prog)
    if shell == "zsh":
        return f"_{stem}"                       # compsys autoloads a file named for its function
    return f"_{stem}_complete" if shell == "bash" else f"__{stem}_complete"


def render_script(shell: str, prog: str) -> str:
    try:
        template = SCRIPTS[shell]
    except KeyError:
        raise CommandError.usage(f"unknown shell {shell!r}; use one of {', '.join(SHELLS)}") from None
    return template.replace("@PROG@", prog).replace("@FUNC@", function_name(shell, prog))
