package assignments.Q1;

public class Student {
    private String name;
    private float percentage;
    private int rollno;
    static int cnt = 100;

    

    public Student(String name, float percentage) {
        this.name = name;
        this.percentage = percentage;
        this.rollno = cnt+1;
        cnt++;
    }

    public String getName() {
        return name;
    }

    public float getPercentage() {
        return percentage;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setPercentage(float percentage) {
        this.percentage = percentage;
    }

    @Override
    public String toString() {
        return "Student [name=" + name + ", percentage=" + percentage + "]";
    }
    
}