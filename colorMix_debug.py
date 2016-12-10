import numpy as np
import numpy.random as rd
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from easygui import multenterbox
from re import findall

class ColorMix:
    
# ********************初始化************************************    
    def __init__(self,WaveRange = (380,730),rgb_ratio = (0.1,0.2,0.7),rgb_info = {"blue":(455,40),"green":(480,50),"red":(560,50)}):
        #对于每一次调色，输入信息就是RGB颜色的峰值波长和半高宽
        self.blackbodyTemp = 7000
        self.flag_rgb = False
        self.flag_Tc = True
        self.rgb_info = rgb_info
        self.rgb_ratio = rgb_ratio
        self.wave_min = WaveRange[0]
        self.wave_max = WaveRange[1]
        self.waveRange = np.array([i for i in range(self.wave_min,self.wave_max+1)],dtype = np.int64)        
# ********************初始化完成************************************

    def blackbody(self):
            T = self.blackbodyTemp
            c1 = 3.7415e20; c2 = 1.43879e7
            planck = lambda x: c1/(x**5)*1/(np.exp(c2/(x*T))-1)
            mixSpec = planck(self.waveRange)
            return {"mixSpec":mixSpec}

# ********************RGB功率谱的生成************************************
    def PowerSpec(self,center,fwhm):
        # max_count为峰值波长处的光子数通量count/second
        # center为峰值波长
        # fwhm为半最大值全宽度
        waveRange = self.waveRange
        # 洛仑兹线型 return 1/(2*np.pi)*fwhm/((waveRange-center)**2+(fwhm/2)**2)
        
        #高斯线型，写成正态分布形式，这样可以直接计算功率，就是在前面加个系数，积分后就是功率
        return 1/(np.sqrt(np.pi)*0.600516*fwhm)*\
               np.exp(-(waveRange-center)**2/(0.600516*fwhm)**2)


    @staticmethod  #颜色填充函数
    def colorFill(x,y):
        visionFunc = np.loadtxt(r"vision.txt") #返回值为数组
        TriFunc = np.loadtxt(r"tristimulus.txt") #观察者三刺激值函数

        #循环移动函数
        def loopMove(x,n):
            num = x.size
            if n>0:
                return np.hstack((x[num-n:],x[:num-n])) #大于0则右移
            else:
                n = abs(n)
                return np.hstack((x[n:],x[num-n:]))
                
        '''
    移动的效果不是很好，还是改变比例系数效果好
        '''
        TriFunc[:,1] = 3.5*loopMove(TriFunc[:,1],0)
        TriFunc[:,2] = 2.4*loopMove(TriFunc[:,2],5)
        TriFunc[:,3] = 2.5*loopMove(TriFunc[:,3],0)
        #plt.plot(x,TriFunc[:,1],"r",x,TriFunc[:,2],"g",x,TriFunc[:,3],"b")
        #plt.show()
        for i in range(x.size-1):
            colorIndex = TriFunc[i,1:] #获取波长对应的颜色坐标
            temp = sum(colorIndex)
            plt.fill( [x[i],x[i+1],x[i+1],x[i],x[i]],[y[i],y[i+1],0,0,y[i]],color =(colorIndex[0]/temp,colorIndex[1]/temp,colorIndex[2]/temp) )


    
    def rgbSpec(self):
        blueSpec = self.rgb_ratio[2]*self.PowerSpec(self.rgb_info["blue"][0],self.rgb_info["blue"][1])
        greenSpec = self.rgb_ratio[1]*self.PowerSpec(self.rgb_info["green"][0],self.rgb_info["green"][1])
        redSpec = self.rgb_ratio[0]*self.PowerSpec(self.rgb_info["red"][0],self.rgb_info["red"][1])
        mixSpec = blueSpec + greenSpec + redSpec #混合谱
        
        greenPower = trapz(greenSpec,self.waveRange)
        redPower = trapz(redSpec,self.waveRange)
        bluePower = trapz(blueSpec,self.waveRange)
        mixPower = trapz(mixSpec,self.waveRange)
        
        rgbPowerRatio = (redPower/mixPower,greenPower/mixPower,bluePower/mixPower)

        #*******************文件输出********************************
       
        np.savetxt("blueSpec.txt",blueSpec,fmt = "%1.4e",newline = "\n")

        
        #*******************文件输出********************************
        
        #绘图
        
        if self.flag_rgb:
            bdSpec = self.blackbody()["mixSpec"]
            #plt.plot(self.waveRange,redSpec,'-or',self.waveRange,greenSpec,'-og',self.waveRange,blueSpec,'-ob')
            #plt.plot(self.waveRange,mixSpec.max()/bdSpec.max()*bdSpec,'k',label = "BlackBody at same ColorLegend")
            #plt.legend()
            fig = plt.figure(figsize = (10,6.18))
            ax = fig.add_subplot(1,1,1)
            ax.plot(self.waveRange,mixSpec,'k',linewidth = 5)
            ax.set_xlabel("Wavelength (nm)",fontsize = 20,weight = 'bold',family = 'serif')
            ax.set_ylabel("Intensity (a.u.)",fontsize = 20,weight = 'bold',family = 'serif')
            ax.set_xticks(np.arange(380,730,30))
            ax.set_yticks(np.arange(0.005,0.040,0.005))
            ax.set_xlim([380,730])
            ax.grid(True)
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(16)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(16)
            fig.subplots_adjust(left = 0.13, right = 0.95, bottom = 0.1, top = 0.95)
            self.colorFill(self.waveRange,mixSpec)
            
            #("(609.296450,11.983115,0.415754)","(540.439943,20.443662,0.562378)","(450,23.234336,0.021868)")
            ax.text(420,0.003,"Pw: 450.00nm\nFHWM: 23.23nm\nPower ratio: 0.022",fontsize = 15)
            ax.text(445,0.015,"Pw: 540.44nm\nFHWM: 20.44nm\nPower ratio: 0.562",fontsize = 15)
            ax.text(622,0.020,"Pw: 609.30nm\nFHWM: 11.99nm\nPower ratio: 0.416",fontsize = 15)
            
            fig.savefig("fig1",dpi = 1200)
            plt.show()

        return {"waveRange":self.waveRange,"mixSpec":mixSpec,"rgbPowerRatio":rgbPowerRatio}        
# ********************调用rgbSpec方法，返回一个spec_info的字典******************

# ********************计算流明和流明效率************************************
    def eff_lumin(self):
        Km = 683 #最大光视效能 683 lm/W
        #vision1是CIE 1924标准，vision是1951年修订版
        visionFunc = np.loadtxt(r"vision1.txt") #返回值为数组
        if not (visionFunc[0][0]==self.wave_min and visionFunc[-1][0]==self.wave_max):
            print("视觉函数与功率谱函数波长维度不匹配")
            raise TypeError("维度不匹配")
        mixSpec = self.rgbSpec()["mixSpec"]
        power = trapz(mixSpec,self.waveRange)
        lumin = trapz(Km*visionFunc[:,1]*mixSpec,self.waveRange)
        luminEff = lumin/power
           
        return {"lumin":lumin,"luminEff":luminEff,"power":power}
# ********************计算流明和流明效率************************************


# ********************计算相对色温和返回待测光源**************************
    def ColorTemp(self):
        TriFunc = np.loadtxt(r"tristimulus.txt") #观察者三刺激值函数
        mixSpec = self.rgbSpec()["mixSpec"]

        #系统的三刺激值
        Xk = trapz(mixSpec*TriFunc[:,1],self.waveRange)
        Yk = trapz(mixSpec*TriFunc[:,2],self.waveRange)
        Zk = trapz(mixSpec*TriFunc[:,3],self.waveRange) 

        #色度坐标
        xk = Xk/(Xk + Yk + Zk); yk = Yk/(Xk + Yk + Zk)
        

        # 计算待测光源的相关色温Tc
        A1 = (xk - 0.3290)/(yk - 0.1870)
        Tc1 = 669*A1**4 - 779*A1**3 + 3660*A1**2 - 7047*A1 + 5652
        #print("Tc1",Tc1)

        A2 = (xk - 0.3316)/(yk - 0.1893)
        Tc2 = 669*A2**4 - 779*A2**3 + 3660*A2**2 - 7047*A2 + 5210
        #print("Tc2",Tc2)

        if (2500<=Tc1<=10000) and (not 10000<=Tc2<=15000):
            Tc = Tc1
        elif (not 2500<=Tc1<=10000) and (10000<=Tc2<=15000):
            Tc = Tc2
        else:
            #raise TypeError("色温计算结果有误")
            Tc = None
            self.flag_Tc = False

        #计算待测光源的CIE 1960 UCS坐标
        uk = 4*xk/(-2*xk+12*yk+3); vk = 6*yk/(-2*xk+12*yk+3)
        
        return {"colorTemp":Tc,"ukvk":(uk,vk),"Yk":Yk}
    
# ********************计算相对色温和返回待测光源UCS坐标**********************

# ********************计算参照光源的参数*************************************
    def sourceRef(self):
        Tc = self.ColorTemp()["colorTemp"]
        #根据相关色温，确定系数group的选择
        paraAll = np.loadtxt(r"SourceRef.txt")
        paraA = paraAll[:,0:3] # 2500K-5000K
        paraB = paraAll[:,3:6] # 5000K-10000K
        paraC = paraAll[:,6:] # 10000K-15000K

        if 2500<=Tc<5000:
            para_RefSource = paraA
        elif 5000<=Tc<10000:
            para_RefSource = paraB
        elif 10000<=Tc<15000:
            para_RefSource = paraC
        else:
            raise TypeError("待测光源相关色温超出数据范围")

        #计算参照光源的UCS坐标和CIE1964颜色空间坐标
        uabc = para_RefSource[0,:]; vabc = para_RefSource[1,:]
        Uabc = para_RefSource[2:10,:]; Vabc = para_RefSource[10:18,:]
        Wabc = para_RefSource[18:,:]
        
        m = 1e4/Tc
        series = np.array([[1,m,m**2]])
        
        ur = list(map(sum,uabc*series))[0]; vr = list(map(sum,vabc*series))[0]

        sum2 = lambda x: sum(x[0,:])
        Ur1 = sum2(Uabc[0,:]*series); Ur2 = sum2(Uabc[1,:]*series)
        Ur3 = sum2(Uabc[2,:]*series); Ur4 = sum2(Uabc[3,:]*series)
        Ur5 = sum2(Uabc[4,:]*series); Ur6 = sum2(Uabc[5,:]*series)
        Ur7 = sum2(Uabc[6,:]*series); Ur8 = sum2(Uabc[7,:]*series)
        
        Vr1 = sum2(Vabc[0,:]*series); Vr2 = sum2(Vabc[1,:]*series)
        Vr3 = sum2(Vabc[2,:]*series); Vr4 = sum2(Vabc[3,:]*series)
        Vr5 = sum2(Vabc[4,:]*series); Vr6 = sum2(Vabc[5,:]*series)
        Vr7 = sum2(Vabc[6,:]*series); Vr8 = sum2(Vabc[7,:]*series)       
        
        Wr1 = sum2(Wabc[0,:]*series); Wr2 = sum2(Wabc[1,:]*series)
        Wr3 = sum2(Wabc[2,:]*series); Wr4 = sum2(Wabc[3,:]*series)
        Wr5 = sum2(Wabc[4,:]*series); Wr6 = sum2(Wabc[5,:]*series)
        Wr7 = sum2(Wabc[6,:]*series); Wr8 = sum2(Wabc[7,:]*series)

        return {"urvr":(ur,vr),"Ur":[Ur1,Ur2,Ur3,Ur4,Ur5,Ur6,Ur7,Ur8],\
                "Vr":[Vr1,Vr2,Vr3,Vr4,Vr5,Vr6,Vr7,Vr8],\
                "Wr":[Wr1,Wr2,Wr3,Wr4,Wr5,Wr6,Wr7,Wr8]}
            
# ********************计算参照光源的参数*************************************

# ********************计算Munsell颜色样品的参数*************************************
    def colorSample(self):
        TriFunc = np.loadtxt(r"tristimulus.txt") #观察者三刺激值函数
        radFactor_5nm = np.loadtxt(r"Munsell.txt")   #颜色样品的辐亮度因子函数
        mixSpec = self.rgbSpec()["mixSpec"]
        
        

        Ycs = {} #颜色样品的系统的Y刺激值
        ucs = {} 
        vcs = {} #颜色样品的CIE1960 USC坐标
        
        for i in range(1,radFactor_5nm.shape[1]):
            # 对辐亮度因子进行插值
            f = interp1d(radFactor_5nm[:,0],radFactor_5nm[:,i])
            radFactor = f(self.waveRange)
            
            Xcsi = trapz(mixSpec*TriFunc[:,1]*radFactor,self.waveRange)
            Ycsi = trapz(mixSpec*TriFunc[:,2]*radFactor,self.waveRange)
            Zcsi = trapz(mixSpec*TriFunc[:,3]*radFactor,self.waveRange)

            Ycs[i] = Ycsi
            ucs[i] = 4*Xcsi/(Xcsi+15*Ycsi+3*Zcsi)
            vcs[i] = 6*Ycsi/(Xcsi+15*Ycsi+3*Zcsi)
        return {"Ycs":Ycs,"ucs":ucs,"vcs":vcs} 
# ********************计算Munsell颜色样品的参数*************************************        
        

# ********************计算参照光源下的CIE 1964颜色空间坐标****************************
    def UVWmod(self):
        func = lambda u,v: (4-u-10*v)/v
        fund = lambda u,v: (1.708*v+0.404-1.481*u)/v

        temp = self.ColorTemp()
        Yk = temp["Yk"]
        # 待测光源调整后的色度坐标
        ck = func(temp["ukvk"][0],temp["ukvk"][1])
        dk = fund(temp["ukvk"][0],temp["ukvk"][1])

        # 参照光源调整后的色度坐标
        ur = self.sourceRef()["urvr"][0]
        vr = self.sourceRef()["urvr"][1]
        cr = func(ur,vr)
        dr = fund(ur,vr)

        funu = lambda ccs,dcs: (10.872+0.404*cr/ck*ccs-4*dr/dk*dcs)/(16.518+1.481*cr/ck*ccs-dr/dk*dcs)
        funv = lambda ccs,dcs: 5.520/(16.518+1.481*cr/ck*ccs-dr/dk*dcs)
        # 颜色样品调整后的色度坐标
        csInfo = self.colorSample()
        Wk = []; Uk = []; Vk = []
        Ycs = csInfo["Ycs"]

        for i in range(len(csInfo["ucs"])):
            ccs = func(csInfo["ucs"][i+1],csInfo["vcs"][i+1])
            dcs = fund(csInfo["ucs"][i+1],csInfo["vcs"][i+1])
            u_cs = funu(ccs,dcs)
            v_cs = funv(ccs,dcs)
            Wk.append(25*(Ycs[i+1]*100/Yk)**(1/3)-17)
            Uk.append(13*Wk[i]*(u_cs-ur))
            Vk.append(13*Wk[i]*(v_cs-vr))

        return {"Wk":Wk,"Uk":Uk,"Vk":Vk}
# ********************计算参照光源下的CIE 1964颜色空间坐标****************************

# ********************计算色差和显色指数****************************
    def Aberration(self):
        WUVk = self.UVWmod(); WUVr = self.sourceRef()
        Wk = np.array(WUVk["Wk"]); Uk = np.array(WUVk["Uk"]); Vk = np.array(WUVk["Vk"])
        Wr = np.array(WUVr["Wr"]); Ur = np.array(WUVr["Ur"]); Vr = np.array(WUVr["Vr"])

        #色差计算
        det_E = np.sqrt((Ur - Uk)**2 + (Vr - Vk)**2 + (Wr - Wk)**2)
        
        #特殊显色指数
        Ri = 100 - 4.6*det_E

        #一般显色指数
        Ra = 1/8*sum(Ri)

        return Ra

# ********************计算色差和显色指数****************************        

def initInput():
    '''
    因为是字典，每次都会随意生成次序，所以没规律，所以只能分开
    values = {"红光LED":"(615,15,0.33)",
               "绿光LED":"(535,45,0.33)",
               "蓝光LED":"(445,20,0.34)"}
    '''
    keys = ("红光LED","绿光LED","蓝光LED")
    values = ("(609.296450,11.983115,0.415754)","(540.439943,20.443662,0.562378)","(450,23.234336,0.021868)")
    inputItem = multenterbox("输入规范：\n"\
                             "(波长，半高宽，光谱功率占比)\n\n"\
                             "<<<请根据输入框的格式更改设置>>>"
                             ,"输入设置参数",keys,values)
    return inputItem


#****************************主函数*********************************
if __name__ == "__main__":

    inputItem = initInput()
    temp = findall(r"(\d+\.?\d*),(\d+\.?\d*),(\d+\.?\d*)",inputItem[0])
    Rled = tuple(map(float,temp[0]))
    temp = findall(r"(\d+\.?\d*),(\d+\.?\d*),(\d+\.?\d*)",inputItem[1])
    Gled = tuple(map(float,temp[0]))
    temp = findall(r"(\d+\.?\d*),(\d+\.?\d*),(\d+\.?\d*)",inputItem[2])
    Bled = tuple(map(float,temp[0]))

    RGB_info = {"blue":Bled[0:-1],"green":Gled[0:-1],"red":Rled[0:-1]}
    colormix = ColorMix(rgb_ratio =(Rled[-1],Gled[-1],Bled[-1]),rgb_info = RGB_info)
    
    
    Tc = colormix.ColorTemp()["colorTemp"]
    Ra = colormix.Aberration()

    
    # 结果输出
    print("\n")
    print("RGB三原色的波长：【%f nm】【%f nm】【%f nm】\n"%(RGB_info["red"][0],RGB_info["green"][0],RGB_info["blue"][0]))
    print("RGB三原色的半高宽：【%f nm】【%f nm】【%f nm】\n"%(RGB_info["red"][1],RGB_info["green"][1],RGB_info["blue"][1]))

    effLumin = colormix.eff_lumin()

    print("待测光源的光谱总功率：%f W\n"%effLumin["power"])
    print("待测光源的流明数：%f lm\n"%effLumin["lumin"])
    print("待测光源的视觉效率：%f lm/W\n"%effLumin["luminEff"])

    colormix.rgbSpec()
    spec = colormix.rgbSpec()
    print("待测光源的RGB三色光谱功率之比：%f:%f:%f\n"%spec["rgbPowerRatio"])
    
    
    print("待测光源的色温：%f K\n" %Tc)
    print("待测光源的显色指数：%f\n" %Ra)
    
    colormix.flag_rgb = True #画图旗帜
    colormix.blackbodyTemp = Tc
    colormix.rgbSpec()

