package Q3;

public class App {
    public static void main(String[] args) {
        Manager m1 = new Manager("Suresh",100000);
        MarketingExecutive mex1 = new MarketingExecutive("Rajesh", 10000, 25);
        System.out.println(m1.display());
        System.out.println(mex1.display());
    }
}

