package Q2;

public class Employee {
    private int empid;
    private String name;
    private Date dob;

    public Employee(int empid, String name, int dd,int mm, int yy) {
        this.empid = empid;
        this.name = name;
        this.dob = new Date(dd,mm,yy);
    }

    public Employee() {
        this.empid = 0;
        this.name = "John Doe";
        this.dob = new Date();
    }

    public int getEmpid() {
        return empid;
    }

    public String getName() {
        return name;
    }

    public Date getDob() {
        return dob;
    }

    public void setEmpid(int empid) {
        this.empid = empid;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setDob(Date dob) {
        this.dob = dob;
    }

    @Override
    public String toString() {
        return "Employee [empid=" + empid + ", name=" + name + ", dob=" + dob.show() + "]";
    }

}
