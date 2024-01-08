package theory.has_a;

public class Employee {
    private int empid;
    private String name;
    private Date dob;

    public Employee() {
        empid=101;
        name="abc";
        dob= new Date();
    }

    public Employee(int empid, String name, int dd, int mm, int yy) {
        this.empid = empid;
        this.name = name;
        this.dob = new Date(dd,mm,yy);
    }

    public void show(){
        System.out.println(this.empid);
        System.out.println(this.name);
        dob.show();
    }
    public static void main(String[] args) {
        Employee e1 = new Employee();
        e1.show();
    }
}
