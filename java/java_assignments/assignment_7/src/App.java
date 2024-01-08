import javax.rmi.CORBA.Util;

public class App {
    public static void main(String[] args) throws Exception {
        UtilityList utl = new UtilityList();
        utl.createList();
        utl.printlist();
        UtilityReport utp = new UtilityReport();
        utp.showReport(utl.getStudents());

    }
}
