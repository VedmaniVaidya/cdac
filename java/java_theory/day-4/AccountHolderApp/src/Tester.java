import java.util.Scanner;

class AccountHolder
{
	private int accno;
	private String name;
	private double balance;
	
	public AccountHolder() {
	}

	public AccountHolder(int accno, String name, double balance) {
		this.accno = accno;
		this.name = name;
		this.balance = balance;
	}
	
	public void deposite(int amount)
	{
		balance=balance+amount;
	}
	
	public void withdraw(int amount)
	{
		balance=balance-amount;
	}

	public String toString() {
		return "AccountHolder [accno=" + accno + ", name=" + name + ", balance=" + balance + "]";
	}

	public int getAccno() {
		return accno;
	}
}


public class Tester {

	public static void main(String[] args) {
			int cnt=0;
			Scanner sc=new Scanner(System.in);
			AccountHolder [] a=new AccountHolder[10];
			
			while(true)
			{
				System.out.println("1. Add a record");
				System.out.println("2. Display all data");
				System.out.println("3. Deposite");
				System.out.println("4. Withdraw");
				System.out.println("5. Exit");
				switch(sc.nextInt())
				{
				case 1:
					System.out.println("Enter info");
					a[cnt++]=new AccountHolder(sc.nextInt(),sc.next(), sc.nextDouble());
					break;
				case 2:
					for(int i=0; i<cnt; i++)
					{
						System.out.println(a[i]);
					}
					break;
				case 3:
					System.out.println("Enter accno to deposite");
					int acno=sc.nextInt();
					System.out.println("Enter amount to be deposited");
					int amt=sc.nextInt();
					for(int i=0; i<cnt; i++)
					{
						if(acno==a[i].getAccno())
						{
							a[i].deposite(amt);
						}
					}
					break;
				case 4:
				case 5:
					System.exit(0);
				}
			}
	}

}



