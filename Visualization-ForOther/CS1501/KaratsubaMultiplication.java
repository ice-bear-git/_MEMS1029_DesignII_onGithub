import java.math.BigInteger;

import java.util.Random;

//cannot use BigInteger predefined methods for multiplication
//cannot use Strings except in computing appropriate exponent
public class KaratsubaMultiplication
{
	private static final BigInteger MAX_INT_VALUE = BigInteger.valueOf(Integer.MAX_VALUE);
	
	public static BigInteger karatsuba(final BigInteger factor0, final BigInteger factor1, final int base)
	{
		//base cases
		// int maxLength = Math.max(factor0.toString().length(), factor1.toString().length());
		// int n = (int)(Math.log(maxLength) / Math.log(2));
		int n = Math.max(factor0.bitLength(), factor1.bitLength());
		if (n <= 1) return factor0.multiply(factor1);
		//downshift to regular multiplication if the factors are both less than the maximum integer values to create a long value
		//we want to divide the number of digits in half (based on the base representation)
		// boolean regularMult =
		// if (factor0.compareTo(MAX_INT_VALUE) < 0 && factor1.compareTo(MAX_INT_VALUE) < 0) {
		// 	regularMult = true;
		// }
		n = n / 2;
		//algorithm
		BigInteger xh = factor0.shiftRight(n);
		BigInteger xl = factor0.subtract(xh.shiftLeft(n));

		BigInteger yh = factor1.shiftRight(n);
		BigInteger yl = factor1.subtract(yh.shiftLeft(n));

		BigInteger m1, m2, m5 = null;
		if (factor0.compareTo(MAX_INT_VALUE) < 0 && factor1.compareTo(MAX_INT_VALUE) < 0) {
			m1 = xh.multiply(yh);
			m2 = xl.multiply(yl);
			m5 = (xh.add(xl)).multiply(yh.add(yl)).subtract(m1).subtract(m2);
		}
		else {
			m1 = karatsuba(xh, yh, base);
			m2 = karatsuba(xl, yl, base);
			m5 = karatsuba(xh.add(xl), yh.add(yl), base).subtract(m1).subtract(m2);
		}

		return m1.shiftLeft(n * 2).add(m5.shiftLeft(n)).add(m2);	
	}


	public static void main(String[] args)
	{
		//test cases
		if(args.length < 3)
		{
			System.out.println("Need two factors and a base value as input");
			return;
		}
		BigInteger factor0 = null;
		BigInteger factor1 = null;
		final Random r = new Random();
		if(args[0].equalsIgnoreCase("r") || args[0].equalsIgnoreCase("rand") || args[0].equalsIgnoreCase("random"))
		{
			factor0 = new BigInteger(r.nextInt(100000), r);
			System.out.println("First factor : " + factor0.toString());
		}
		else
		{
			factor0 = new BigInteger(args[0]);
		}
		if(args[1].equalsIgnoreCase("r") || args[1].equalsIgnoreCase("rand") || args[1].equalsIgnoreCase("random"))
		{
			factor1 = new BigInteger(r.nextInt(100000), r);
			System.out.println("Second factor : " + factor1.toString());
		}
		else
		{
			factor1 = new BigInteger(args[1]);
		}
		final BigInteger result = karatsuba(factor0, factor1, Integer.parseInt(args[2]));
		System.out.println(result);
		System.out.println(result.equals(factor0.multiply(factor1)));
	}
}