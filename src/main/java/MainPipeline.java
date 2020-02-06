import edu.wpi.first.vision.VisionPipeline;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.opencv.core.CvType.CV_8UC3;

public class MainPipeline implements VisionPipeline {
    private GripPipeline grip = new GripPipeline();

    private Mat debugMat = new Mat(240, 320, CV_8UC3, new Scalar(0, 0, 0));
    private Optional<Target> target = Optional.empty();
    private Mat currentFrame;

    private static final double DIST_TO_HEIGHT_RATIO = Math.tan((Math.PI / 180) * (34.3/2)) * (130.0 / 115.0);

    @Override
    public void process(Mat mat) {
        currentFrame = mat;
        grip.process(mat);

        debugMat.setTo(new Scalar(0, 0, 0));
        ArrayList<MatOfPoint2f> conts = getContours();
        target = pullBest(conts).map((r) -> {
            Point center;
            {
                Pair<Point, Point> l = getTopLine(r);
                center = new Point((l.getLeft().x + l.getRight().x) / 2, (l.getLeft().y + l.getRight().y) / 2);
                Imgproc.circle(debugMat, center, 3, new Scalar(0, 255, 0), -1);
            }

            double dist;
            {
                drawRR(debugMat, r);

                Pair<Point, Point> ll = getLeftLine(r);
                Imgproc.line(debugMat, ll.getLeft(), ll.getRight(), new Scalar(110, 100, 125), 2);

                double ld = distance(ll);
                dist = (240*17)/(2 * ld * DIST_TO_HEIGHT_RATIO);
            }

            return new Target(dist, center);
        });

        ArrayList<MatOfPoint> conts2 = conts.stream().map(MainPipeline::makei).collect(Collectors.toCollection(ArrayList::new));
        Imgproc.drawContours(debugMat, conts2, -1, new Scalar(255, 255, 255), 1);
    }

    public ArrayList<MatOfPoint2f> getContours() {
        ArrayList conts = grip.filterContoursOutput();
        for (int i = 0; i < conts.size(); i++) {
            conts.set(i, approxPolyDP((MatOfPoint) conts.get(i)));
        }
        return (ArrayList<MatOfPoint2f>) conts;
    }

    public void writeDebug(Mat matOut) {
        debugMat.copyTo(matOut);
    }

    public Optional<Target> getTarget() {
        return target;
    }

    public Mat getCurrentFrame() {
        return currentFrame;
    }

    public static void drawRR(Mat m, RotatedRect r) {
        if (r == null) {
            return;
        }
        Point[] ps = new Point[4];
        r.points(ps);
        for (int i = 0; i < 4; i++) {
            Imgproc.line(m, ps[i], ps[(i + 1) & 3], new Scalar(255, 0, 0), 1);
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

    public static double distance(Pair<Point, Point> p) {
        return distance(p.getLeft(), p.getRight());
    }

    public static double distance(Point p1, Point p2) {
        double xd = p1.x - p2.x;
        double yd = p1.y - p2.y;
        return Math.sqrt(xd*xd + yd*yd);
    }

    public static Optional<RotatedRect> pullBest(ArrayList<MatOfPoint2f> conts) {
        return conts.stream()
                .map((v) -> new Pair<>(v, Imgproc.minAreaRect(v)))
                .max(Comparator.comparingDouble((p) -> rrArea(p.getRight()) / Imgproc.contourArea(p.getLeft())))
                .map(Pair::getRight);
    }

    public static double rrArea(RotatedRect in) {
        Point[] pts = new Point[4];
        in.points(pts);
        return Imgproc.contourArea(new MatOfPoint2f(pts));
    }

    public static Pair<Point, Point> getTopLine(RotatedRect in) {
        Point[] pts = new Point[4];
        in.points(pts);
        Arrays.sort(pts, Comparator.comparingDouble(v -> v.y));
        return new Pair<>(pts[0], pts[1]);
    }

    public static Pair<Point, Point> getLeftLine(RotatedRect in) {
        Point[] pts = new Point[4];
        in.points(pts);
        Arrays.sort(pts, Comparator.comparingDouble(v -> v.x));
        return new Pair<>(pts[0], pts[1]);
    }
}