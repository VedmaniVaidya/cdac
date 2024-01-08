import java.util.Scanner;

class AccountHolder {
    static Scanner sc = new Scanner(System.in);
    static int num_accounts = 0;
    static AccountHolder[] accounts = new AccountHolder[1000];
    
    private int account_number;
    private String name;
    private float balance;

    public AccountHolder(int account_number, String name, float balance) {
        this.account_number = account_number+1;
        this.name = name;
        this.balance = balance;
        num_accounts+=1;
    }

    public AccountHolder() {
        this.account_number = 123123123;
        this.name = "abc";
        this.balance = 100000;
    }
    
    //getters
    public int getAccount_number() {
        return account_number;
    }

    public String getName() {
        return name;
    }

    public float getBalance() {
        return balance;
    }
    //setters
    public void setAccount_number(int account_number) {
        this.account_number = account_number;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setBalance(float balance) {
        this.balance = balance;
    }
    
    public void deposit(float amount){
        this.balance+=amount;
        System.out.println(this.balance);
    }

    public void withdraw(float amount){
        this.balance-=amount;
        System.out.println(this.balance);
    }


    @Override
    public String toString() {
        return "[account_number=" + account_number + ", name=" + name + ", balance=" + balance + "]";
    }


    public static void create_account(){
        System.out.println("Enter account holders name:");
        String name = sc.nextLine();
        accounts[AccountHolder.num_accounts] = new AccountHolder(num_accounts,name,0);
    }
    
    
}

public class Q5 {
    public static void main(String[] args) {
        menu();

    }

    public static void menu(){
        int choice;
        int flag=0;
        while(flag!=-1){
        Scanner sc = new Scanner(System.in);
        System.out.println("Select choice:\n"+//
                "1. Add record for account holder\n" + //
                "2. Display details of all account holders.\n" + //
                "3. Deposite an amount into perticular account\n" + //
                "4. Withdraw an amount from perticular account");
        choice = sc.nextInt();
        switch (choice) {
            case 1:

                AccountHolder.create_account();
                break;

            case 2:
                System.out.println("Details of all accounts");
                for (int i=0;i<AccountHolder.num_accounts;i++){
                    System.out.println(AccountHolder.accounts[i]);
                }
                break;

            case 3:
                System.out.println("Enter your account number for deposit");
                int acc_num = sc.nextInt();
                for(int i=0;i<AccountHolder.num_accounts;i++){
                    if(AccountHolder.accounts[i].getAccount_number()==acc_num){
                        System.out.println("Enter how much amount you want to deposit");
                        float amount = sc.nextFloat();
                        AccountHolder.accounts[i].deposit(amount);
                    }
                if (i==AccountHolder.num_accounts-1){
                    System.out.println("Account not found");
                }
                }    
                    break;
                case 4:
                    System.out.println("Enter your account number for withdrawl");
                    acc_num = sc.nextInt();
                    for(int i=0;i<AccountHolder.num_accounts;i++){
                        if(AccountHolder.accounts[i].getAccount_number()==acc_num){
                            System.out.println("Enter how much amount you want to wihtdraw");
                            float amount = sc.nextFloat();
                            AccountHolder.accounts[i].withdraw(amount);
                        }
                    if(i==AccountHolder.num_accounts-1){
                        System.out.println("Account not found");
                    } 
                    }    
                    break;

                case 5:
                    flag = -1;
                    break;
            default:
            System.out.println("Invalid choice");
                break;
        }

    }
}

}