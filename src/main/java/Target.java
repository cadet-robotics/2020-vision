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

    /*
    public double getHorizontalAngle() {
        double diff = center.x - MainPipeline.WIDTH;
        double sig = Math.signum(diff);
        diff *= sig;
        return Math.asin(diff * Math.sin(MainPipeline.H_FOV / 2) / MainPipeline.WIDTH * 2) * sig;
    }
    */
}