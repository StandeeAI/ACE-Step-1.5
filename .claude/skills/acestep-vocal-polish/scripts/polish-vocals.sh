#!/bin/bash
# Separate vocals from an ACE-Step track with demucs, treat them with ffmpeg, remix.
# Usage: polish-vocals.sh <input_audio> [--preset subtle|vintage|custom] [--out FILE]
#        [--vocal-gain 1.65] [--music-gain 0.85] [--echo 0.14] [--lufs -15.5] [--vocals-out FILE]
#        [--vocals-in FILE --music-in FILE]   use stems separated earlier instead of running demucs
#        custom only: [--low-db 2] [--mid-db -1.5] [--presence-db -3] [--air-db 0] [--comp-threshold -24]
#                     [--comp-ratio 3.5] [--echo-delay 70] [--noise 0.02]   (the defaults are the subtle preset)
# --vocals-out also writes the treated vocals alone (same chain, gain and tape wobble; not loudness-normalised).
set -euo pipefail

PRESET="subtle"; OUT=""; VOCALS_OUT=""; VOCAL_GAIN=""; MUSIC_GAIN="0.85"; ECHO=""; LUFS="-15.5"
VOCALS_IN=""; MUSIC_IN=""
LOW_DB="2"; MID_DB="-1.5"; PRESENCE_DB="-3"; AIR_DB="0"; COMP_THRESHOLD="-24"; COMP_RATIO="3.5"; ECHO_DELAY_ARG="70"; NOISE_ARG="0.02"
IN="${1:-}"
[ -n "$IN" ] && [ -f "$IN" ] || { sed -n '2,8p' "$0"; exit 1; }
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --preset) PRESET="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --vocal-gain) VOCAL_GAIN="$2"; shift 2 ;;
        --music-gain) MUSIC_GAIN="$2"; shift 2 ;;
        --echo) ECHO="$2"; shift 2 ;;
        --lufs) LUFS="$2"; shift 2 ;;
        --vocals-out) VOCALS_OUT="$2"; shift 2 ;;
        --vocals-in) VOCALS_IN="$2"; shift 2 ;;
        --music-in) MUSIC_IN="$2"; shift 2 ;;
        --low-db) LOW_DB="$2"; shift 2 ;;
        --mid-db) MID_DB="$2"; shift 2 ;;
        --presence-db) PRESENCE_DB="$2"; shift 2 ;;
        --air-db) AIR_DB="$2"; shift 2 ;;
        --comp-threshold) COMP_THRESHOLD="$2"; shift 2 ;;
        --comp-ratio) COMP_RATIO="$2"; shift 2 ;;
        --echo-delay) ECHO_DELAY_ARG="$2"; shift 2 ;;
        --noise) NOISE_ARG="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
{ [ -z "$VOCALS_IN" ] && [ -z "$MUSIC_IN" ]; } || { [ -f "$VOCALS_IN" ] && [ -f "$MUSIC_IN" ]; } \
    || { echo "--vocals-in and --music-in go together and must exist" >&2; exit 1; }

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

case "$PRESET" in
    # Tuned by ear: warm low-mids, cut 5.5 kHz harshness, one short quiet echo.
    subtle)
        VOX_EQ="highpass=f=110,lowpass=f=9500,acompressor=threshold=-24dB:ratio=3.5:makeup=3,equalizer=f=250:t=q:w=1:g=2,equalizer=f=2800:t=q:w=1.2:g=-1.5,equalizer=f=5500:t=q:w=1.5:g=-3,asoftclip=type=tanh:param=1.2"
        : "${VOCAL_GAIN:=1.65}"; : "${ECHO:=0.14}"; ECHO_DELAY="70"
        MUSIC_FX="volume=${MUSIC_GAIN}"; NOISE_AMP="0.02"; WOW="0.03" ;;
    # Narrow and saturated; stacked echoes sound hollow, so keep levels modest.
    vintage)
        VOX_EQ="highpass=f=220,lowpass=f=7000,acompressor=threshold=-26dB:ratio=5:makeup=3,asoftclip=type=tanh:param=2"
        : "${VOCAL_GAIN:=1.8}"; : "${ECHO:=0.2}"; ECHO_DELAY="90"
        MUSIC_FX="lowpass=f=11000,asoftclip=type=tanh:param=1.4,volume=${MUSIC_GAIN}"; NOISE_AMP="0.04"; WOW="0.06" ;;
    # The subtle chain with every setting open (the mixer in the studio UI); with the defaults it is the subtle preset.
    custom)
        AIR=""; [ "$AIR_DB" = "0" ] || AIR=",highshelf=f=10000:g=${AIR_DB}"
        VOX_EQ="highpass=f=110,lowpass=f=9500,acompressor=threshold=${COMP_THRESHOLD}dB:ratio=${COMP_RATIO}:makeup=3,equalizer=f=250:t=q:w=1:g=${LOW_DB},equalizer=f=2800:t=q:w=1.2:g=${MID_DB},equalizer=f=5500:t=q:w=1.5:g=${PRESENCE_DB}${AIR},asoftclip=type=tanh:param=1.2"
        : "${VOCAL_GAIN:=1.65}"; : "${ECHO:=0.14}"; ECHO_DELAY="$ECHO_DELAY_ARG"
        MUSIC_FX="volume=${MUSIC_GAIN}"; NOISE_AMP="$NOISE_ARG"; WOW="0.03" ;;
    *) echo "Unknown preset: $PRESET" >&2; exit 1 ;;
esac

BASE="$(basename "${IN%.*}")"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
[ -n "$OUT" ] || OUT="$(dirname "$IN")/polished-${PRESET}_${BASE}.mp3"
DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")"

if [ -z "$VOCALS_IN" ]; then
    DEMUCS="$(command -v demucs || echo "$HOME/.local/bin/demucs")"
    [ -x "$DEMUCS" ] || { echo "demucs missing: uv tool install demucs --python 3.11 --with numpy --with soundfile" >&2; exit 1; }
    # </dev/null keeps demucs/ffmpeg from consuming the stdin of any calling loop.
    "$DEMUCS" --two-stems=vocals -o "$WORK" --mp3 "$IN" </dev/null >/dev/null 2>&1
    VOCALS_IN="$WORK/htdemucs/$BASE/vocals.mp3"; MUSIC_IN="$WORK/htdemucs/$BASE/no_vocals.mp3"
fi

# With --vocals-out the treated vocals are split off before the mix and get the same tape wobble and top cut as the mix.
VOX_TAIL="[v];"; VOCALS_MAP=()
if [ -n "$VOCALS_OUT" ]; then
    VOX_TAIL=",asplit=2[v][vs];[vs]vibrato=f=0.6:d=${WOW},lowpass=f=14000[vo];"
    VOCALS_MAP=(-map "[vo]" -q:a 2 "$VOCALS_OUT")
fi

ffmpeg -v error -y -i "$VOCALS_IN" -i "$MUSIC_IN" \
    -f lavfi -i "anoisesrc=color=pink:amplitude=${NOISE_AMP}:duration=${DUR}" -filter_complex "
[0:a]${VOX_EQ},aecho=0.85:0.7:${ECHO_DELAY}:${ECHO},volume=${VOCAL_GAIN}${VOX_TAIL}
[1:a]${MUSIC_FX}[b];
[2:a]highpass=f=2000,lowpass=f=9000,volume=0.25[n];
[b][v][n]amix=inputs=3:normalize=0:duration=first,vibrato=f=0.6:d=${WOW},lowpass=f=14000,loudnorm=I=${LUFS}:TP=-1.5[o]" \
    -map "[o]" -q:a 2 "$OUT" ${VOCALS_MAP[@]+"${VOCALS_MAP[@]}"} </dev/null
echo "$OUT"
