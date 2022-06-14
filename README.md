# Tracking-and-Recognition
Была реализована система трекинга и распознавания лиц. Проведено тестирование методов детектирования, распознавания и трекинга лиц. В итоговой системе ля распознавания лиц используется библиотека facenet-pytorch, для детектирования используется MTCNN, для трекинга используется метод ByteTrack.

## Описание содержимого файлов

* ssd_detection.py, mtcnn_detection.py: содержат тестирование методов детектирования лиц
* celeba.py, chokepoint.py, masked.py, facenet_celeba_embedding.py, facenet_chokepoint_embedding.py, masked_embedding.py, face_recognition.py, face_recognition_embedding.py: содержат получение шаблонных векторов признаков и тестирование методов распознавания лиц
* chokepoint_annot.py: содержит преобразование аннотации с координатами глаз к аннотации с координатами ограничивающих лица прямоугольников
* system.py: содержит код с применением всех выбранных методов
* bytetrack.py, sort.py: содержат тестирование методов трекинга лиц

![](diagram.png)

