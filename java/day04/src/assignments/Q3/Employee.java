package assignments.Q3;

public class Employee {
    private int empid;
    private String name;
    private float salary;
    static int cnt = 1;


    public Employee(String name, float salary) {
        this.empid = cnt++;
        this.name = name;
        this.salary = salary;
    }


    public Employee() {
        this.empid = cnt++;
        this.name = "John Doe";
        this.salary = 20000;

    }


    @Override
    public String toString() {
        return "Employee [empid=" + empid + ", name=" + name + ", salary=" + salary + " ";
    }


    public int getEmpid() {
        return empid;
    }


    public String getName() {
        return name;
    }


    public float getSalary() {
        return salary;
    }


    public void setEmpid(int employeeId) {
        this.empid = employeeId;
    }


    public void setName(String name) {
        this.name = name;
    }


    public void setSalary(float salary) {
        this.salary = salary;
    }


}
