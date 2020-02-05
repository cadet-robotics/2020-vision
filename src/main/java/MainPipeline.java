import edu.wpi.first.vision.VisionPipeline;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.opencv.core.CvType.CV_8UC3;

public class MainPipeline implements VisionPipeline {
    private GripPipeline grip = new GripPipeline();

    @Override
    public void process(Mat mat) {
        grip.process(mat);
    }

    public ArrayList<MatOfPoint2f> getContours() {
        ArrayList conts = grip.filterContoursOutput();
        for (int i = 0; i < conts.size(); i++) {
            conts.set(i, approxPolyDP((MatOfPoint) conts.get(i)));
        }
        return (ArrayList<MatOfPoint2f>) conts;
    }

    public void writeDebug(Mat matOut) {
        Mat m = new Mat(240, 320, CV_8UC3, new Scalar(0, 0, 0));
        ArrayList<MatOfPoint2f> conts = getContours();
        pullBest(conts).ifPresent((v) -> {
            drawRR(m, v);
        });
        ArrayList<MatOfPoint> conts2 = conts.stream().map(MainPipeline::makei).collect(Collectors.toCollection(ArrayList::new));
        Imgproc.drawContours(m, conts2, -1, new Scalar(255, 255, 255), 1);
        m.copyTo(matOut);
        m.release();
    }

    public static void drawRR(Mat m, RotatedRect r) {
        if (r == null) {
            return;
        }
        Point[] ps = new Point[4];
        r.points(ps);
        for (int i = 0; i < 4; i++) {
            Imgproc.line(m, ps[i], ps[(i + 1) & 3], new Scalar(255, 0, 0), 2);
        }
    }

    public static MatOfPoint2f make2f(MatOfPoint pIn) {
        return new MatOfPoint2f(pIn.toArray());
    }

    public static MatOfPoint makei(MatOfPoint2f pIn) {
        return new MatOfPoint(pIn.toArray());
    }

    public static MatOfPoint2f approxPolyDP(MatOfPoint inS) {
        MatOfPoint2f in = make2f(inS);
        double e = 0.01 * Imgproc.arcLength(in, true);
        MatOfPoint2f out = new MatOfPoint2f();
        Imgproc.approxPolyDP(in, out, e, true);
        return out;
    }

    public static Optional<RotatedRect> pullBest(ArrayList<MatOfPoint2f> conts) {
        return conts.stream()
                .map((v) -> new Pair<>(v, Imgproc.minAreaRect(v)))
                .max((p1, p2) -> {
                    double v1 = rrArea(p1.getRight()) / Imgproc.contourArea(p1.getLeft());
                    double s1 = Math.abs((v1 - 2.7) / 2.7);
                    double v2 = rrArea(p2.getRight()) / Imgproc.contourArea(p2.getLeft());
                    double s2 = Math.abs((v2 - 2.7) / 2.7);
                    if (s1 < s2) {
                        return 1;
                    } else {
                        return -1;
                    }
                })
                .map(Pair::getRight);
    }

    public static double rrArea(RotatedRect in) {
        Point[] pts = new Point[4];
        in.points(pts);
        return Imgproc.contourArea(new MatOfPoint2f(pts));
    }
}

class Pair<T, V> {
    private T left;
    private V right;

    public Pair(T leftIn, V rightIn) {
        left = leftIn;
        right = rightIn;
    }

    public T getLeft() {
        return left;
    }

    public V getRight() {
        return right;
    }
}