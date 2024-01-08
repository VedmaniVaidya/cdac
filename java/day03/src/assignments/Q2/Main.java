package assignments.Q2;

import assignments.Q2.com.cdac.pack1.Student;
import assignments.Q2.com.cdac.pack2.Batch;

public class Main {
    public static void main(String[] args) {
        Student student1 = new Student(1, "abc");
        Batch batch1 = new Batch("cs", 40);
        System.out.println(student1+" "+batch1);
    }
}
