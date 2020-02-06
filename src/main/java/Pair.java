public class Pair<T, V> {
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