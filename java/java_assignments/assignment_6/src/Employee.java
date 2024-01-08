import java.util.Scanner;

public class Employee {
    static Scanner sc = new Scanner(System.in);
    private String emp_id;
    private String name;
    private float salary;

    public Employee() {
        this.emp_id = "default";
        this.name = "default";
        this.salary = 0;
    }

    public Employee(String emp_id, String name, float salary) {
        this.emp_id = emp_id;
        this.name = name;
        this.salary = salary;
    }

    public String getEmp_id() {
        return emp_id;
    }

    public String getName() {
        return name;
    }

    public float getSalary() {
        return salary;
    }

    public void setEmp_id(String emp_id) {
        this.emp_id = emp_id;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setSalary(float salary) {
        this.salary = salary;
    }

    

    @Override
    public String toString() {
        return "emp_id=" + emp_id + ", name=" + name + ", salary=" + salary+"\n";
    }

    public static Employee add_emp(){
        Employee obj = new Employee();
        System.out.println("Enter Employee ID");
        obj.setEmp_id(sc.next());
        System.out.println("Enter Employee Name");
        obj.setName(sc.next());
        System.out.println("Enter Employee Salary");
        obj.setSalary(sc.nextFloat());
        return obj;
    }

    public void show_employee(){
        System.out.println(this.toString());
    }
    
    public void update_info(String name, float salary){
        this.setName(name);
        this.setSalary(salary);
    }
}
