package upcasting;

public class Upcasting {
    public static void main(String[] args) {
        Employee e1 = new WageEmployee(1, "Jack", 10, 10, 2010, 10, 700);
        e1.show();
    }
}
