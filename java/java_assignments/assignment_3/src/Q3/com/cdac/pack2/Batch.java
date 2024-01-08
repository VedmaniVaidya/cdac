package Q3.com.cdac.pack2;

public class Batch {
    private String courseName;
    private int batchstrength;
    public Batch(String courseName, int batchstrength) {
        this.courseName = courseName;
        this.batchstrength = batchstrength;
    }
    @Override
    public String toString() {
        return "Batch [courseName=" + courseName + ", batchstrength=" + batchstrength + "]";
    }

}
