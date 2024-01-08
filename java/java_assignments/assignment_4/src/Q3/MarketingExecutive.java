package Q3;

public class MarketingExecutive extends Manager {
    private float kms_travelled;
    static int tal_perkm = 5;
    static int tal_phone = 2000;


    public MarketingExecutive(String name, float basicsalary, float kmstravelled) {
        super(name, basicsalary);
        this.kms_travelled = kmstravelled;
    }

    public float gross_salary(){
        return super.getSalary() + ((MarketingExecutive.tal_perkm*this.kms_travelled) + MarketingExecutive.tal_phone);
    }

    public float net_salary(){
        float pf = 0.125f * super.getSalary();
        return this.gross_salary() - pf;
    }

    public String display(){
        return super.toString() + " gross_salary: "+this.gross_salary()+ " net_salary: "+this.net_salary() + "]";
    }


}
