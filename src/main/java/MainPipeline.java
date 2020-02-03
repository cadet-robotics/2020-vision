import edu.wpi.first.vision.VisionPipeline;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

import static org.opencv.core.CvType.CV_8UC3;

public class MainPipeline implements VisionPipeline {
    private GripPipeline grip = new GripPipeline();

    @Override
    public void process(Mat mat) {
        grip.process(mat);
    }

    public void writeDebug(Mat matOut) {
        Mat m = new Mat(240, 320, CV_8UC3, new Scalar(0, 0, 0));
        ArrayList<MatOfPoint> conts = grip.filterContoursOutput();
        for (int i = 0; i < conts.size(); i++) {
            drawRR(m, Imgproc.fitEllipse(new MatOfPoint2f(conts.get(i))));
            Imgproc.drawContours(m, conts, 0, new Scalar(255, 255, 255), 1);
        }
        m.copyTo(matOut);
        m.release();
    }

    public static void drawRR(Mat m, RotatedRect r) {
        Point[] ps = new Point[4];
        r.points(ps);
        for (int i = 0; i < 4; i++) {
            Imgproc.line(m, ps[i], ps[(i + 1) & 3], new Scalar(255, 0, 0), 2);
        }
    }
}