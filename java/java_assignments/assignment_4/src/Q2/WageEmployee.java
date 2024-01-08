package Q2;

public class WageEmployee extends Employee{
    private int num_hrs;
    private int rate;

    public WageEmployee(int empid, String name, int dd, int mm, int yy, int num_hrs, int rate) {
        super(empid, name, dd, mm, yy);
        this.num_hrs = num_hrs;
        this.rate = rate;
    }
    public WageEmployee(int num_hrs, int rate) {
        super();
        this.num_hrs = num_hrs;
        this.rate = rate;
    }

    public WageEmployee() {
        super();
        this.num_hrs = 0;
        this.rate = 0;
    }


    public int getNum_hrs() {
        return num_hrs;
    }
    public int getRate() {
        return rate;
    }
    public void setNum_hrs(int num_hrs) {
        this.num_hrs = num_hrs;
    }
    public void setRate(int rate) {
        this.rate = rate;
    }
    @Override
    public String toString() {
        return super.toString()+" "+"WageEmployee [num_hrs=" + num_hrs + ", rate=" + rate +"]";
    }

    public float calculate_salary(){
        float sal=0;
        sal = this.getNum_hrs()*this.getRate();
        return sal;
    }

}
