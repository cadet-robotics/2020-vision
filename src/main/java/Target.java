import org.opencv.core.Point;

public class Target {
    // inches
    private double dist;
    private Point center;

    public Target(double distIn, Point centerIn) {
        dist = distIn;
        center = centerIn;
    }

    public double getDist() {
        return dist;
    }

    public Point getCenter() {
        return center;
    }
}