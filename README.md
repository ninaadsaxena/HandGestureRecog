# Hand Gesture Recognition System

A real-time hand gesture recognition system built using Python, OpenCV, and MediaPipe. This project recognizes various hand gestures such as **Thumbs Up**, **Open Palm**, **Peace**, and **Rock On** with real-time feedback displayed on a video feed. The project was developed using **PyCharm**.

## Features

- **Real-Time Gesture Detection**: Recognizes gestures in real-time using a webcam feed.
- **Multiple Gesture Support**:
  - Thumbs Up
  - Open Palm
  - Peace Sign
  - Rock On Sign
- **Handedness Detection**: Identifies whether the detected hand is left or right.
- **Stylish UI**: Displays gesture name and FPS with a clean overlay.
- **Efficient Performance**: Optimized for accuracy and speed using MediaPipe's Hand Solution.

## Prerequisites

Ensure you have the following installed on your system:

- **Python 3.7+**
- **OpenCV**
- **MediaPipe**
- **NumPy**

Install the required libraries using:
```bash
pip install opencv-python mediapipe numpy
```

## How It Works

1. The script uses MediaPipe to detect hand landmarks.
2. Custom logic processes the landmarks to classify gestures based on finger positions.
3. The recognized gesture and hand type (left/right) are displayed on the video feed.

### Supported Gestures

1. **Thumbs Up**: Thumb is extended while other fingers are bent.
2. **Open Palm**: All fingers are extended.
3. **Peace Sign**: Index and middle fingers extended, others bent.
4. **Rock On Sign**: Thumb and little finger extended, others bent.

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. Run the script:
   ```bash
   python HandRec2.py
   ```

3. Press `q` to quit the application.

## Development Environment

- **Code Editor**: PyCharm
- **Language**: Python

## Customization

- **Add New Gestures**: Modify the `recognize_gesture` method to define new gesture conditions based on landmarks.
- **Change Video Dimensions**: Adjust `cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)` and `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ...)` for your desired resolution.

## Example Output

- **Thumbs Up**: Displays "Right Hand: Thumbs Up" on the screen.
- **Peace Sign**: Displays "Left Hand: Peace" on the screen.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request. For significant changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/) for providing efficient hand tracking and landmark detection.
- [OpenCV](https://opencv.org/) for video processing.

## Author

Developed by **Ninaad Saxena**. Feel free to connect on [LinkedIn](https://www.linkedin.com/in/ninaadsaxena/).

---
