class MyThread implements Runnable
{
	Thread t1, t2;
	
	public MyThread(){
		t1=new Thread(this,"t1");
		t2=new Thread(this,"t2");
		t1.start();
		t2.start();
	}
	
	public void run() {
			for(int i=0; i<1000; i++)
			{
				if(Thread.currentThread()==t1)
					System.out.println("first thread..");
				else if(Thread.currentThread()==t2)
					System.out.println("second thread..");
			}
	}
}
public class Tester {
	public static void main(String[] args) {
		new MyThread();
	}
}
