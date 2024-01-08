import java.util.Scanner;
import java.util.Vector;

public class App {
    public static void main(String[] args) throws Exception {
        Scanner input = new Scanner(System.in);
        Vector<Employee> v = new Vector<>();
        // v.add(Employee.add_user());
        // System.out.println(v.elementAt(0));
        boolean flag = true;
        while (true) {
            System.out.println("Select choice\n"+
            "a. Insert record into a Vector.\n"+
            "b. Update information of a specific employee on the basis of emp_id accepted from user.\n"+
            "c. Display all the records.\n");
            String choice = input.next();

            switch (choice) {
                case "a":
                    v.add(Employee.add_emp());
                    break;
                
                case "b":
                    System.out.println("Enter Employee ID to update employee information");
                    String emp_id = input.next();
                    for (Employee obj:v){
                        boolean x = false;
                        if (emp_id.equals(obj.getEmp_id())){
                            System.out.println("Enter new name");
                            String name=input.next();
                            obj.setName(name);
                            System.out.println("Enter new salary");
                            float salary=input.nextFloat();
                            obj.setSalary(salary);
                            System.out.println("Updated Employee Information");
                            obj.show_employee();
                            x = true;
                        }
                        else{
                            x=false;
                        }
                    if (x==false){
                        System.out.println("Employee id not found\nTry again");
                    }
                    }
                    break;
                
                case "c":
                    System.out.println(v);
                    break;
            
                default:
                    System.out.println("Invalid choice. Try again");
                    break;
            }
        }
    }
}
