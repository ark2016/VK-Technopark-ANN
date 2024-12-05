# Результаты 

Лучше всего показала себя `nyrahealth/CrisperWhisper`, модель занимающая лидирующие позиции в leaderboard по ASR на HuggingFace.

У дркгих архитектур ожидаемо хуже результаты: wav2vec2, даже зафайнтюненая на русский язык (`jonatasgrosman/wav2vec2-large-xlsr-53-russian`) не смогла даже близко подойти к `openai/whisper-large-v2`, не говоря о `openai/whisper-large-v3` и `openai/whisper-large-v3-turbo`, которые показали достойный резльтат.

`openai/whisper-large-v3-turbo` превзошёл только файнтюн и 2 место `nyrahealth/CrisperWhisper`

Пробовал также запустить `nvidia/canary-1b`, но были сложности с библиотеками и не хватило времени полностью разрешить конфликты с различными версиями модулей(
