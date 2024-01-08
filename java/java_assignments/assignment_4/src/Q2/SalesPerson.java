package Q2;

public class SalesPerson extends WageEmployee {
    private int num_items_sold;
    private int commision;
    
    public SalesPerson(int empid, String name, int dd, int mm, int yy, int num_hrs, int rate, int num_items_sold,
            int commision) {
        super(empid, name, dd, mm, yy, num_hrs, rate);
        this.num_items_sold = num_items_sold;
        this.commision = commision;
    }
    public SalesPerson(int num_hrs, int rate, int num_items_sold, int commision) {
        super(num_hrs, rate);
        this.num_items_sold = num_items_sold;
        this.commision = commision;
    }

    public SalesPerson() {
        super();
        this.num_items_sold = 0;
        this.commision = 0;
    }

    
    public int getNum_items_sold() {
        return num_items_sold;
    }
    public int getCommision() {
        return commision;
    }

    @Override
    public float calculate_salary() {
        float sal=0;
        sal = super.calculate_salary()+this.getNum_items_sold()*this.getCommision();
        return sal;
    }
    @Override
    public String toString() {
        return super.toString()+" "+"SalesPerson [num_items_sold=" + num_items_sold + ", commision=" + commision + "]";
    }

    
    
}
