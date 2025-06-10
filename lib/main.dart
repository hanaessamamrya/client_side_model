import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:video_player/video_player.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:gal/gal.dart'; 

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Soccer Ball Detection',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: SoccerBallDetectionScreen(),
    );
  }
}

class SoccerBallDetectionScreen extends StatefulWidget {
  const SoccerBallDetectionScreen({super.key});

  @override
  _SoccerBallDetectionScreenState createState() =>
      _SoccerBallDetectionScreenState();
}

class _SoccerBallDetectionScreenState extends State<SoccerBallDetectionScreen> {
  // Controllers for text input
  final TextEditingController _inputPathController = TextEditingController();
  final TextEditingController _outputPathController = TextEditingController();

  // Video player controller
  VideoPlayerController? _videoController;
  VideoPlayerController? _outputVideoController; // Add this

  // TensorFlow Lite interpreter
  Interpreter? _interpreter;

  // Processing state
  bool _isProcessing = false;
  String _status = 'Ready';
  double _progress = 0.0;

  // Square corners - now interactive
  final List<Offset> _corners = [];
  int _currentCornerIndex = 0;
  bool _isCornerSelectionMode = false;

  // Video dimensions for coordinate mapping
  Size? _videoSize;

  // Model input/output details
  static const int INPUT_SIZE = 640;
  static const double CONFIDENCE_THRESHOLD = 0.5;
  static const double IOU_THRESHOLD = 0.4;
  static const int SOCCER_BALL_CLASS_ID = 32;

  // For displaying output video
  String? _outputVideoPath;
  // For elapsed time
  Duration? _elapsedTime;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _outputVideoController?.dispose(); // Dispose output controller
    _interpreter?.close();
    _inputPathController.dispose();
    _outputPathController.dispose();
    super.dispose();
  }

  // Load YOLOv5 model
  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/yolo11n_float32.tflite',
      );
      setState(() {
        _status = 'Model loaded successfully';
      });
    } catch (e) {
      setState(() {
        _status = 'Error loading model: $e';
      });
    }
  }

  // Pick input video file
  Future<void> _pickInputVideo() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
    );

    if (result != null) {
      _inputPathController.text = result.files.single.path!;
      _initializeVideoPlayer();
    }
  }


  // Initialize video player
  Future<void> _initializeVideoPlayer() async {
    if (_inputPathController.text.isEmpty) return;

    _videoController?.dispose();
    _videoController = VideoPlayerController.file(
      File(_inputPathController.text),
    );

    try {
      await _videoController!.initialize();
      _videoSize = _videoController!.value.size;
      setState(() {
        _corners.clear();
        _currentCornerIndex = 0;
        _isCornerSelectionMode = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error initializing video: $e';
      });
    }
  }

  // Start corner selection mode
  void _startCornerSelection() {
    setState(() {
      _corners.clear();
      _currentCornerIndex = 0;
      _isCornerSelectionMode = true;
      _status = 'Tap corner 1 on the video';
    });
  }

  // Handle tap on video for corner selection
  void _handleVideoTap(TapDownDetails details) {
    if (!_isCornerSelectionMode || _videoController == null) return;

    // Calculate the actual video display size within the container
    final Size containerSize = Size(
      MediaQuery.of(context).size.width - 32, // Accounting for padding
      200, // Video container height
    );

    final Size videoSize = _videoController!.value.size;
    final double videoAspectRatio = videoSize.width / videoSize.height;
    final double containerAspectRatio =
        containerSize.width / containerSize.height;

    Size displaySize;
    Offset displayOffset = Offset.zero;

    if (videoAspectRatio > containerAspectRatio) {
      // Video is wider - fit to width
      displaySize = Size(
        containerSize.width,
        containerSize.width / videoAspectRatio,
      );
      displayOffset = Offset(
        0,
        (containerSize.height - displaySize.height) / 2,
      );
    } else {
      // Video is taller - fit to height
      displaySize = Size(
        containerSize.height * videoAspectRatio,
        containerSize.height,
      );
      displayOffset = Offset((containerSize.width - displaySize.width) / 2, 0);
    }

    // Convert tap position to video coordinates
    // Convert tap position to video coordinates
    final Offset tapPosition = details.localPosition;
    final Offset adjustedTap = Offset(
      tapPosition.dx - displayOffset.dx,
      tapPosition.dy - displayOffset.dy,
    );

    // Check if tap is within the video display area
    if (adjustedTap.dx >= 0 &&
        adjustedTap.dx <= displaySize.width &&
        adjustedTap.dy >= 0 &&
        adjustedTap.dy <= displaySize.height) {
      // Convert to actual video pixel coordinates
      final double scaleX = videoSize.width / displaySize.width;
      final double scaleY = videoSize.height / displaySize.height;

      final Offset videoCoordinate = Offset(
        adjustedTap.dx * scaleX,
        adjustedTap.dy * scaleY,
      );

      setState(() {
        _corners.add(videoCoordinate);
        _currentCornerIndex++;

        if (_currentCornerIndex < 4) {
          _status = 'Tap corner ${_currentCornerIndex + 1} on the video';
        } else {
          _isCornerSelectionMode = false;
          _status = 'All 4 corners selected successfully!';
        }
      });
    }
  }

  // Clear all selected corners
  void _clearCorners() {
    setState(() {
      _corners.clear();
      _currentCornerIndex = 0;
      _isCornerSelectionMode = false;
      _status = 'Corners cleared';
    });
  }

  // Process video with YOLO detection
  Future<void> _processVideo() async {
    if (_interpreter == null) {
      setState(() {
        _status = 'Model not loaded';
      });
      return;
    }

    if (_inputPathController.text.isEmpty) {
      setState(() {
        _status = 'Please specify input video path';
      });
      return;
    }

    if (_corners.length != 4) {
      setState(() {
        _status = 'Please select all 4 corners first';
      });
      return;
    }

    setState(() {
      _isProcessing = true;
      _status = 'Processing video...';
      _elapsedTime = null;
    });

    final Stopwatch stopwatch = Stopwatch()..start();

    try {
      // Set output path automatically
      final dir = await getApplicationDocumentsDirectory();
      _outputVideoPath = '${dir.path}/output_video.mp4';

      await _performVideoProcessing();
      stopwatch.stop();

      // Initialize output video controller for playback
      _outputVideoController?.dispose();
      _outputVideoController = VideoPlayerController.file(File(_outputVideoPath!));
      await _outputVideoController!.initialize();

      // Automatically save to gallery
      await Gal.putVideo(_outputVideoPath!);

      setState(() {
        _status = 'Video processing completed and saved to gallery!';
        _elapsedTime = stopwatch.elapsed;
      });

      // Optionally show a snackbar
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Saved to gallery!')),
      );
    } catch (e) {
      stopwatch.stop();
      debugPrint("Error during video processing: $e");
      setState(() {
        _status = 'Error processing video: $e';
        _elapsedTime = stopwatch.elapsed;
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  // Perform actual video processing
  Future<void> _performVideoProcessing() async {
    final String inputPath = _inputPathController.text;
    final String outputPath = _outputVideoPath!;

    // Create temporary directory for frames
    final Directory tempDir = await getTemporaryDirectory();
    final String framesDir = '${tempDir.path}/frames';
    final String processedFramesDir = '${tempDir.path}/processed_frames';

    await Directory(framesDir).create(recursive: true);
    await Directory(processedFramesDir).create(recursive: true);

    try {
      // Step 1: Extract frames from video
      setState(() {
        _status = 'Extracting frames from video...';
        _progress = 0.1;
      });

      await _extractFramesFromVideo(inputPath, framesDir);

      // Step 2: Process each frame with YOLO
      setState(() {
        _status = 'Processing frames with YOLO...';
        _progress = 0.3;
      });

      await _processFramesWithYOLO(framesDir, processedFramesDir);

      // Step 3: Reconstruct video from processed frames
      setState(() {
        _status = 'Reconstructing video...';
        _progress = 0.8;
      });

      await _reconstructVideoFromFrames(
        processedFramesDir,
        outputPath,
        inputPath,
      );

      // Step 4: Cleanup temporary files
      setState(() {
        _status = 'Cleaning up...';
        _progress = 0.95;
      });

      await Directory(framesDir).delete(recursive: true);
      await Directory(processedFramesDir).delete(recursive: true);

      setState(() {
        _progress = 1.0;
      });
    } catch (e) {
      debugPrint("Error during video processing: $e");
      // Cleanup on error
      try {
        if (await Directory(framesDir).exists()) {
          await Directory(framesDir).delete(recursive: true);
        }
        if (await Directory(processedFramesDir).exists()) {
          await Directory(processedFramesDir).delete(recursive: true);
        }
      } catch (_) {}
      rethrow;
    }
  }

  // Extract frames from video using FFmpeg
  Future<void> _extractFramesFromVideo(
    String inputPath,
    String framesDir,
  ) async {
    final String command =
        '-i "$inputPath" -vf fps=30 "$framesDir/frame_%06d.png"';

    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();

    if (!ReturnCode.isSuccess(returnCode)) {
      final logs = await session.getLogs();
      throw Exception(
        'Failed to extract frames: ${logs.map((log) => log.getMessage()).join('\n')}',
      );
    }
  }

  // Process frames with YOLO detection
  Future<void> _processFramesWithYOLO(
    String framesDir,
    String processedFramesDir,
  ) async {
    final Directory framesDirObj = Directory(framesDir);
    final List<FileSystemEntity> frameFiles = framesDirObj
        .listSync()
        .where((file) => file.path.endsWith('.png'))
        .toList();

    frameFiles.sort((a, b) => a.path.compareTo(b.path));

    for (int i = 0; i < frameFiles.length; i++) {
      final File frameFile = File(frameFiles[i].path);
      final String frameName = frameFile.path.split('/').last;

      // Load and process frame
      final Uint8List frameBytes = await frameFile.readAsBytes();
      final img.Image? image = img.decodeImage(frameBytes);

      if (image != null) {
        // Run YOLO inference
        final List<Detection> detections = await _runYOLOInference(image);

        // Draw overlays (square and detections)
        final img.Image processedImage = _drawOverlays(image, detections);

        // Save processed frame
        final String processedFramePath = '$processedFramesDir/$frameName';
        final Uint8List processedBytes = Uint8List.fromList(
          img.encodePng(processedImage),
        );
        await File(processedFramePath).writeAsBytes(processedBytes);
      }

      // Update progress
      setState(() {
        _progress = 0.3 + (0.5 * (i + 1) / frameFiles.length);
        _status = 'Processing frame ${i + 1}/${frameFiles.length}';
      });
    }
  }

  // Reconstruct video from processed frames
  Future<void> _reconstructVideoFromFrames(
    String processedFramesDir,
    String outputPath,
    String inputPath,
  ) async {
    // Get original video framerate
    String getFramerateCommand =
        '-i "$inputPath" -hide_banner 2>&1 | grep -o -P "(?<=, )[0-9]+(?= fps)"';

    // Default framerate if we can't detect it
    String framerate = '30';

    final String command =
        '-framerate $framerate -i "$processedFramesDir/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p -y "$outputPath"';

    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();

    if (!ReturnCode.isSuccess(returnCode)) {
      final logs = await session.getLogs();
      throw Exception(
        'Failed to reconstruct video: ${logs.map((log) => log.getMessage()).join('\n')}',
      );
    }
  }

  // Run YOLO inference on image
  Future<List<Detection>> _runYOLOInference(img.Image image) async {
    if (_interpreter == null) return [];

    // Resize image to model input size
    final img.Image resizedImage = img.copyResize(
      image,
      width: INPUT_SIZE,
      height: INPUT_SIZE,
    );

    // Convert to float32 and normalize
    final Float32List input = Float32List(INPUT_SIZE * INPUT_SIZE * 3);
    int pixelIndex = 0;

    for (int y = 0; y < INPUT_SIZE; y++) {
      for (int x = 0; x < INPUT_SIZE; x++) {
        final pixel = resizedImage.getPixel(x, y);
        double r, g, b;

        // For newer image package versions (Color object)
        r = pixel.r.toDouble();
        g = pixel.g.toDouble();
        b = pixel.b.toDouble();

        input[pixelIndex++] = r / 255.0;
        input[pixelIndex++] = g / 255.0;
        input[pixelIndex++] = b / 255.0;
      }
    }

    // Reshape input for batch dimension
    final inputTensor = input.reshape([1, INPUT_SIZE, INPUT_SIZE, 3]);

    // Prepare output tensor
   final outputTensor = List.generate(1, (_) => List.generate(84, (_) => List.filled(8400, 0.0)));

    // Run inference
    _interpreter!.run(inputTensor, outputTensor);

    // Post-process output
    final List<Detection> detections = _postProcessOutput(
      outputTensor[0],
      image.width,
      image.height,
    );

    print(_interpreter!.getOutputTensors()[0].shape);

    return detections;
  }

  // Post-process YOLO output
  List<Detection> _postProcessOutput(
    List<List<double>> output, // output shape: [84][8400]
    int originalWidth,
    int originalHeight,
  ) {
    List<Detection> detections = [];

    // Transpose output: [84][8400] -> [8400][84]
    for (int i = 0; i < output[0].length; i++) {
      List<double> detection = List.generate(output.length, (j) => output[j][i]);

      final double centerX = detection[0];
      final double centerY = detection[1];
      final double width = detection[2];
      final double height = detection[3];
      final double confidence = detection[4];

      if (confidence < CONFIDENCE_THRESHOLD) continue;

      final List<double> classScores = detection.sublist(5);

      double maxScore = 0;
      int classId = -1;
      for (int j = 0; j < classScores.length; j++) {
        if (classScores[j] > maxScore) {
          maxScore = classScores[j];
          classId = j;
        }
      }

      if (classId != SOCCER_BALL_CLASS_ID) continue;

      final double finalConfidence = confidence * maxScore;
      if (finalConfidence < CONFIDENCE_THRESHOLD) continue;

      final double scaleX = originalWidth / INPUT_SIZE;
      final double scaleY = originalHeight / INPUT_SIZE;

      final double x1 = (centerX - width / 2) * scaleX;
      final double y1 = (centerY - height / 2) * scaleY;
      final double x2 = (centerX + width / 2) * scaleX;
      final double y2 = (centerY + height / 2) * scaleY;

      detections.add(
        Detection(
          bbox: [x1, y1, x2, y2],
          confidence: finalConfidence,
          classId: classId,
        ),
      );
    }

    return _applyNMS(detections);
  }

  // Apply Non-Maximum Suppression
  List<Detection> _applyNMS(List<Detection> detections) {
    if (detections.isEmpty) return [];

    // Sort detections by confidence (descending)
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<Detection> result = [];
    List<bool> suppressed = List.filled(detections.length, false);

    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;

      result.add(detections[i]);

      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;

        double iou = _calculateIoU(detections[i].bbox, detections[j].bbox);
        if (iou > IOU_THRESHOLD) {
          suppressed[j] = true;
        }
      }
    }

    return result;
  }

  // Calculate Intersection over Union
  double _calculateIoU(List<double> box1, List<double> box2) {
    final double x1 = math.max(box1[0], box2[0]);
    final double y1 = math.max(box1[1], box2[1]);
    final double x2 = math.min(box1[2], box2[2]);
    final double y2 = math.min(box1[3], box2[3]);

    if (x2 <= x1 || y2 <= y1) return 0.0;

    final double intersection = (x2 - x1) * (y2 - y1);
    final double box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    final double box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    final double union = box1Area + box2Area - intersection;

    return intersection / union;
  }

  // Draw overlays on image
  img.Image _drawOverlays(img.Image image, List<Detection> detections) {
    final img.Image result = img.Image.from(image);

    // Draw square from corners
    if (_corners.length == 4) {
      _drawSquare(result);
    }

    // Draw detection bounding boxes
    for (Detection detection in detections) {
      _drawBoundingBox(result, detection);
    }

    return result;
  }

  // Draw square on image
  void _drawSquare(img.Image image) {
    for (int i = 0; i < 4; i++) {
      final Offset start = _corners[i];
      final Offset end = _corners[(i + 1) % 4];

      img.drawLine(
        image,
        x1: start.dx.round(),
        y1: start.dy.round(),
        x2: end.dx.round(),
        y2: end.dy.round(),
        color: img.ColorFloat64.rgb(0, 255, 0), // Green color
        thickness: 3,
      );
    }
  }

  // Draw bounding box on image
  void _drawBoundingBox(img.Image image, Detection detection) {
    final int x1 = detection.bbox[0].round();
    final int y1 = detection.bbox[1].round();
    final int x2 = detection.bbox[2].round();
    final int y2 = detection.bbox[3].round();

    // Draw rectangle
    img.drawRect(
      image,
      x1: x1,
      y1: y1,
      x2: x2,
      y2: y2,
      color: img.ColorFloat32.rgb(255, 0, 0), // Red color
      thickness: 2,
    );

    // Draw confidence text
    final String confidenceText =
        '${(detection.confidence * 100).toStringAsFixed(1)}%';
    img.drawString(
      image,
      confidenceText,
      font: img.arial24,
      x: x1,
      y: y1 - 15,
      color: img.ColorFloat32.rgb(255, 255, 255),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Soccer Ball Detection'),
        backgroundColor: Colors.green,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status indicator
            Container(
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _isProcessing ? Colors.orange[100] : Colors.green[100],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                children: [
                  Text(
                    'Status: $_status',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: _isProcessing
                          ? Colors.orange[800]
                          : Colors.green[800],
                    ),
                  ),
                  if (_isProcessing) ...[
                    SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: _progress,
                      backgroundColor: Colors.grey[300],
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.orange),
                    ),
                    SizedBox(height: 4),
                    Text(
                      '${(_progress * 100).toStringAsFixed(1)}%',
                      style: TextStyle(fontSize: 12, color: Colors.orange[800]),
                    ),
                  ],
                  if (_elapsedTime != null) ...[
                    SizedBox(height: 8),
                    Text(
                      // Format as mm:ss.mmm
                      'Elapsed time: ${_elapsedTime!.inMinutes}:${(_elapsedTime!.inSeconds % 60).toString().padLeft(2, '0')}.${(_elapsedTime!.inMilliseconds % 1000).toString().padLeft(3, '0')} minutes',
                      style: TextStyle(fontSize: 14, color: Colors.black87),
                    ),
                  ],
                ],
              ),
            ),
            SizedBox(height: 20),

            // Input video path
            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Input Video Path',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: TextField(
                            controller: _inputPathController,
                            decoration: InputDecoration(
                              hintText: 'Select input video file',
                              border: OutlineInputBorder(),
                            ),
                          ),
                        ),
                        SizedBox(width: 8),
                        ElevatedButton(
                          onPressed: _pickInputVideo,
                          child: Text('Browse'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            // Corner selection interface
            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Corner Selection',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed:
                                _videoController != null && !_isProcessing
                                ? _startCornerSelection
                                : null,
                            icon: Icon(Icons.touch_app),
                            label: Text('Select Corners'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.blue,
                              foregroundColor: Colors.white,
                            ),
                          ),
                        ),
                        SizedBox(width: 8),
                        ElevatedButton.icon(
                          onPressed: _corners.isNotEmpty && !_isProcessing
                              ? _clearCorners
                              : null,
                          icon: Icon(Icons.clear),
                          label: Text('Clear'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.red,
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Selected corners (${_corners.length}/4):',
                      style: TextStyle(fontWeight: FontWeight.w500),
                    ),
                    for (int i = 0; i < _corners.length; i++)
                      Padding(
                        padding: EdgeInsets.only(left: 16, top: 4),
                        child: Text(
                          'Corner ${i + 1}: (${_corners[i].dx.round()}, ${_corners[i].dy.round()})',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),

            // Video preview with overlay
            if (_videoController != null &&
                _videoController!.value.isInitialized)
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Video Preview${_isCornerSelectionMode ? ' - Tap to select corners' : ''}',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: _isCornerSelectionMode
                              ? Colors.orange
                              : Colors.black,
                        ),
                      ),
                      SizedBox(height: 8),
                      GestureDetector(
                        onTapDown: _handleVideoTap,
                        child: SizedBox(
                          height: 200,
                          width: double.infinity,
                          child: Stack(
                            children: [
                              VideoPlayer(_videoController!),
                              if (_corners.isNotEmpty && _videoSize != null)
                                CustomPaint(
                                  size: Size(double.infinity, 200),
                                  painter: CornerOverlayPainter(
                                    corners: _corners,
                                    videoSize: _videoSize!,
                                    containerSize: Size(
                                      MediaQuery.of(context).size.width - 64,
                                      200,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          IconButton(
                            onPressed: () {
                              setState(() {
                                _videoController!.value.isPlaying
                                    ? _videoController!.pause()
                                    : _videoController!.play();
                              });
                            },
                            icon: Icon(
                              _videoController!.value.isPlaying
                                  ? Icons.pause
                                  : Icons.play_arrow,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

            SizedBox(height: 20),

            // Process button
            ElevatedButton(
              onPressed: _isProcessing ? null : _processVideo,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isProcessing
                  ? Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.white,
                            ),
                          ),
                        ),
                        SizedBox(width: 8),
                        Text('Processing...', style: TextStyle(fontSize: 16)),
                      ],
                    )
                  : Text(
                      'Process Video',
                      style: TextStyle(fontSize: 16, color: Colors.white),
                    ),
            ),
            SizedBox(height: 20),

            // Output video preview (playable, no manual save button)
            if (_outputVideoPath != null && File(_outputVideoPath!).existsSync())
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Output Video Preview',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 8),
                      if (_outputVideoController != null &&
                          _outputVideoController!.value.isInitialized)
                        AspectRatio(
                          aspectRatio: _outputVideoController!.value.aspectRatio,
                          child: VideoPlayer(_outputVideoController!),
                        ),
                      if (_outputVideoController != null &&
                          _outputVideoController!.value.isInitialized)
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            IconButton(
                              onPressed: () {
                                setState(() {
                                  _outputVideoController!.value.isPlaying
                                      ? _outputVideoController!.pause()
                                      : _outputVideoController!.play();
                                });
                              },
                              icon: Icon(
                                _outputVideoController!.value.isPlaying
                                    ? Icons.pause
                                    : Icons.play_arrow,
                              ),
                            ),
                          ],
                        ),
                      SizedBox(height: 12),
                      Text(
                        'Video automatically saved to gallery.',
                        style: TextStyle(color: Colors.green[800]),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          if (_videoController != null) {
            _videoController!.seekTo(Duration.zero);
          }
        },
        tooltip: 'Reset Video',
        child: Icon(Icons.refresh),
      ),
    );
  }
}

class Detection {
  final List<double> bbox; // [x1, y1, x2, y2]
  final double confidence;
  final int classId;

  Detection({
    required this.bbox,
    required this.confidence,
    required this.classId,
  });
}

// Custom painter for corner overlay
class CornerOverlayPainter extends CustomPainter {
  final List<Offset> corners;
  final Size videoSize;
  final Size containerSize;

  CornerOverlayPainter({
    required this.corners,
    required this.videoSize,
    required this.containerSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (corners.isEmpty) return;

    // Calculate scaling and offset (same as in _handleVideoTap)
    final double videoAspectRatio = videoSize.width / videoSize.height;
    final double containerAspectRatio =
        containerSize.width / containerSize.height;

    Size displaySize;
    Offset displayOffset = Offset.zero;

    if (videoAspectRatio > containerAspectRatio) {
      displaySize = Size(
        containerSize.width,
        containerSize.width / videoAspectRatio,
      );
      displayOffset = Offset(
        0,
        (containerSize.height - displaySize.height) / 2,
      );
    } else {
      displaySize = Size(
        containerSize.height * videoAspectRatio,
        containerSize.height,
      );
      displayOffset = Offset((containerSize.width - displaySize.width) / 2, 0);
    }

    final double scaleX = displaySize.width / videoSize.width;
    final double scaleY = displaySize.height / videoSize.height;

    final Paint paint = Paint()
      ..color = Colors.green.withOpacity(0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    // Draw lines between selected corners
    if (corners.isNotEmpty) {
      Path path = Path();
      for (int i = 0; i < corners.length; i++) {
        final Offset c = Offset(
          corners[i].dx * scaleX + displayOffset.dx,
          corners[i].dy * scaleY + displayOffset.dy,
        );
        if (i == 0) {
          path.moveTo(c.dx, c.dy);
        } else {
          path.lineTo(c.dx, c.dy);
        }
      }
      if (corners.length == 4) {
        path.close();
      }
      canvas.drawPath(path, paint);

      // Draw circles at each corner
      for (int i = 0; i < corners.length; i++) {
        final Offset c = Offset(
          corners[i].dx * scaleX + displayOffset.dx,
          corners[i].dy * scaleY + displayOffset.dy,
        );
        canvas.drawCircle(c, 6, Paint()..color = Colors.green);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
