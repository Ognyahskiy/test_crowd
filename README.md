# Crowd Detection

Небольшой проект для детекции людей на видео с помощью **YOLO**.  
Работает в докере, можно запускать с GPU и сразу сохранять результат в локальную папку.

---

## Как запустить

Собрать образ:

```bash
docker build -t crowd-detect .
```
Положить видео в папку и запустить:
```
docker run \
  -v "$(pwd)":/app \
  crowd-detect \
  --name crowd.mp4 \
  --model n
```
Если есть видеокарта — можно ускорить:
```
docker run \
  --gpus all \
  -v "$(pwd)":/app \
  crowd-detect \
  --name crowd.mp4 \
  --model n
```
Аргументы

--name — видеофайл

--model — размер модели YOLO (n, s, m, l, x)
