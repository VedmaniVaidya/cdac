package theory.has_a;

public class Date {
    private int dd,mm,yy;

    public Date(){
        dd=1;
        mm=1;
        yy=2000;
    }

    public Date(int dd, int mm, int yy) {
        this.dd = dd;
        this.mm = mm;
        this.yy = yy;
    }

    public void show(){
    System.out.println(dd+"/"+mm+"/"+yy);
    }

}
