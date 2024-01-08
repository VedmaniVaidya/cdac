package theory;
import theory.has_a.Employee;
import theory.is_a_inheritance.WageEmployee;
public class App {
    public static void main(String[] args) throws Exception {
        // System.out.println("Hello, World!");
        Employee e1 = new Employee(102,"jay",23,10,1990);
        e1.show();
        WageEmployee e2 = new WageEmployee();
        e2.show();
        
        //downcasting
        //Employee e3 = new WageEmployee();
        //e3.show();
    }
}