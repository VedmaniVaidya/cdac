public class VarArgs {
	
    public static int add(int ...a) {
		int sum1= 0;
		for(int i=0;i<a.length;i++) {
			sum1+=i;
		}
		return sum1;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int sum = add(1,2,3,4,5);
		System.out.println(sum);
		System.out.println(add(2,4,6,8));

	}

}
