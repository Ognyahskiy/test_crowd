import argparse
import cv2

from ultralytics import YOLO


def main():
    """
    Обрабатывает входной видеоролик, выполняя детекцию людей с помощью модели YOLO,
    и сохраняет выходной ролик с отрисованными результатами.

    Функция:
      • Принимает путь к входному видеофайлу через аргумент командной строки `--name`.
      • Загружает модель YOLO выбранного размера (n, s, m, l, x) через аргумент `--model`.
      • Выполняет кадр за кадром детекцию класса «person» (ID 0).
      • Рисует рамки и подписи поверх кадров.
      • Сохраняет результат в файл `crowd_result.mp4` и отображает его в реальном времени.

    Аргументы командной строки:
        --name (str): Путь к входному видеофайлу. Обязательный аргумент.
        --model (str): Размер модели YOLO (n, s, m, l, x). По умолчанию — "x".

    Вывод:
        Сохраняет видеофайл `crowd_result.mp4` с нанесёнными боксами.
        В процессе обработки показывает окно с результатом.
        Для выхода — нажать клавишу `q`.

    Пример:
        $ python main.py --name crowd.mp4 --model s
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Видеофайл")
    parser.add_argument("--model", type=str, default="x", help="Выбор модели yolo:n,s,m,l,x")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.name)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter("crowd_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    model = YOLO(f"yolo11{args.model}.pt")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model(frame, classes=[0], show=False, verbose=False)
            for res in results[0].boxes:
                x1, y1, x2, y2 = map(int, res.xyxy[0].cpu().numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "person", (x2 - 65, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "person", (x2 - 65, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(res.conf[0].cpu().numpy().round(2)), (x2, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str(res.conf[0].cpu().numpy().round(2)), (x2, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0, 0, 0), 1, cv2.LINE_AA)
            writer.write(frame)


if __name__ == "__main__":
    main( )
