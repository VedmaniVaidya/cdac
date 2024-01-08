class Account
{
	private int balance;
	
	public Account(){
		balance=10000;
	}
	
	public void deposite(int amount)
	{
		///////code here
		synchronized(this)
		{
			System.out.println("balance before deposite- "+balance);
			balance=balance+amount;
			System.out.println("balance after deposite- "+balance);
		}
		//////code here
	}
	
	public void withdraw(int amount)
	{
		//code here
		synchronized(this)
		{
			System.out.println("balance before withdraw- "+balance);
			balance=balance-amount;
			System.out.println("balance after withdraw- "+balance);
		}
		//code here
	}
}

class AccountUser extends Thread
{
	private String name, tot;
	private Account account;
	private int amount;
	public AccountUser(String n, Account acc, String tran, int amt)
	{
		name=n;
		account=acc;
		tot=tran;
		amount=amt;
	}
	
	public void run(){
		if(tot.equals("deposite"))
			account.deposite(amount);
		else if(tot.equals("withdraw"))
			account.withdraw(amount);
	}
}

public class Tester {

	public static void main(String[] args) {
		Account account=new Account();
		AccountUser user1=new AccountUser("abc",account,"deposite",1000);
		AccountUser user2=new AccountUser("xyz",account,"withdraw",500);
		user1.start();
		user2.start();
	}
}





