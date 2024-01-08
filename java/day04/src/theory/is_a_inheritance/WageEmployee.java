package theory.is_a_inheritance;
import theory.has_a.Employee;

public class WageEmployee extends Employee{
    private int hours;
    private int rate;

    public WageEmployee() {
        this.hours = 10;
        this.rate = 100;
    }

    public WageEmployee(int hours, int rate) {
        this.hours = hours;
        this.rate = rate;
    }
    public WageEmployee(int empid, String name, int dd, int mm, int yy, int hours, int rate) {
        super(empid, name, dd, mm, yy);
        this.hours = hours;
        this.rate = rate;
    }

    public void show(){
        super.show();
        System.out.println(hours);
        System.out.println(rate);
    }

    
}
