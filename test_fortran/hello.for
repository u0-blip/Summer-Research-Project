        PROGRAM SUBDEM 
        REAL A,B,C,SUM,SUMSQ 
        CALL INPUT( + A,B,C)
        CALL CALC(A,B,C,SUM,SUMSQ)
        CALL OUTPUT(SUM,SUMSQ)
        END

        SUBROUTINE INPUT(X, Y, Z)
        REAL X,Y,Z
        PRINT *,'ENTER THREE NUMBERS => '
        READ *,X,Y,Z
        RETURN
        END
c       What is going on
        SUBROUTINE CALC(A,B,C, 
     &SUM,SUMSQ)
        REAL A,B,C,SUM,SUMSQ
        SUM = A + B + C
        SUMSQ = SUM **2
        RETURN
        END

        SUBROUTINE OUTPUT(SUM,SUMSQ)
        REAL SUM, SUMSQ
        PRINT *, 'Ah, the humanity'
        PRINT *,'The sum of the numbers you entered are: ',SUM
        PRINT *,'And the square of the sum is:',SUMSQ
        RETURN
        END
