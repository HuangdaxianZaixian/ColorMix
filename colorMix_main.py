'''
光源的光谱采用高斯函数，化成连续正态分布函数形式，系数即功率

1924 CIE明视觉 vision1.txt
1956 CIE明视觉修订版 vision.txt
三刺激值 tristimulus.txt
参照光源参数 sourceRef.txt
孟塞尔颜色样品 Munsell.txt

所有谱函数的波长范围均为【380nm,730nm】
波长的单位全部用nm
最大色温范围：2500K-15000K（可调用其他宽色温范围经验函数）

标准黑体辐射验证

迭代标准：
在规定的条件下，寻找光效最高的配色方案


计算流程：
                                                                            

                    待测光源部分                                                         参照照明体部分                                          孟塞尔标准颜色样品
                                                                 
                                                                  
                                                                  
    1. 生成待测光源的功率密度光谱，计算其功率                          1. 以待测光源色温为参照照明体色温Tc,m = 1e4/Tc          1. 确定孟塞尔1-8号标准颜色样品的光谱辐亮度因子谱函数数据   
                       ↓                                                                     ↓
                       ↓                                                                     ↓
    2. 根据明视觉谱数据,计算待测光源的流明数和视能光效                  2. 确定参照照明体系数表，根据 f = a + b*m +c*m**2,
                       ↓                                                 计算参照照明体统一的色度坐标（ur,vr）,并计算前
                       ↓                                                 8个样品的(U*,V*,W*)
    3. 根据CIE 1931 标准色度学系统的三刺激值，结合待测光源                                      ↓
       的的功率密度谱，计算待测光源的色度坐标（xk,yk),                                          ↓
       并根据公式，将其转换成 CIE 1960 UCS坐标 （uk,vk）               3. 计算待测光源与照明体的色度差，判断是否满足要求
                       ↓                                                                     ↓
                       ↓                                                                     ↓
    4. 根据分段经验公式，计算待测光源的色温                            4. (U*,V*,W*)即孟塞尔颜色样品在参照照明体的颜色数据
                       ↓
                       ↓
    5. 确定孟塞尔1-8号标准颜色样品的光谱辐亮度因子谱函数数据
                       ↓
                       ↓
    6. 计算孟塞尔1-8标准颜色样品在待测光源下的色度坐标（xki,yki）
       和CIE 1960 UCS坐标（uki,vki）
                       ↓
                       ↓
    7. 将待测光源下的色坐标转换到参照照明体下的色坐标，即适应性
       色位移，获得相应的CIE 1964 颜色空间坐标（U*ki,V*ki,W*ki）
                                                              
                                                                 ↓↓
                                                                 ↓↓
                                                            显色指数的计算
误差来源：
1. 色温的经验公式，参照照明体的参数计算需要用到待测光源的色温
2. 孟塞尔颜色样品光谱辐亮度因子标准数据的非连续性，插值误差（5nm间隔）
3. 非连续谱

检验途径：
1. 生成谱的验证
2. 生成特定色温下的黑体辐射谱，黑体显指为100
3. 比对文献数据


改进：
1. 加入不同荧光粉，不同激发波长下的激发谱数据
   荧光粉激发谱（峰值波长、半高宽、谱功率）跟激发（功率、峰值波长、半高宽）的关系
2. LED的EQE数据
                                                       
'''
import numpy as np
import numpy.random as rd
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from easygui import multchoicebox,multenterbox,boolbox
from re import compile,findall

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
        self.waveRange = np.array([i for i in range(self.wave_min,self.wave_max+1)],dtype = np.float64)        
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
        #绘图
        
        if self.flag_rgb:
            bdSpec = self.blackbody()["mixSpec"]
            plt.plot(self.waveRange,redSpec,'-or',self.waveRange,greenSpec,'-og',self.waveRange,blueSpec,'-ob')
            plt.plot(self.waveRange,mixSpec,'k',linewidth = 5)
            plt.plot(self.waveRange,mixSpec.max()/bdSpec.max()*bdSpec,'k',label = "BlackBody at same ColorLegend")
            plt.legend()
            self.colorFill(self.waveRange,mixSpec)

            #plt.text(650,0.005,"Blackbody")
            plt.show()

        return {"waveRange":self.waveRange,"mixSpec":mixSpec,"rgbPowerRatio":rgbPowerRatio}        
# ********************调用rgbSpec方法，返回一个spec_info的字典******************

# ********************计算流明和流明效率************************************
    def eff_lumin(self):
        Km = 683 #最大光视效能 683 lm/W
        visionFunc = np.loadtxt(r"vision.txt") #返回值为数组
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
def initSet():
    setItem = multchoicebox("默认设置：\n" \
                            "红光LED >>> \t波长范围 (615nm-650nm) \t半高宽范围 (10nm-20nm) \t光谱功率占比范围 (0.0-1.0)\n"\
                            "绿光LED >>> \t波长范围 (500nm-560nm) \t半高宽范围 (30nm-60nm) \t光谱功率占比范围 (0.0-1.0)\n"\
                            "蓝光LED >>> \t波长范围 (445nm-465nm) \t半高宽范围 (15nm-30nm) \t光谱功率占比范围 (0.0-1.0)\n"\
                            "色温范围>>> \t最大范围：(2500K-15000K)\n"
                            "显色指数>>> \t最低要求：70\n"\
                            "迭代次数>>> \t10000 \n\n"
                            "<请选择需要重新设置的项目>(多选)",\
                            "设置参数选择",\
                            ("红光LED","绿光LED","蓝光LED","色温范围","显色指数","迭代次数"))
    return setItem
        
                            

def initInput(setItem):
    default = {"红光LED":"(615,650)+(10,20)+(0.0,1.0)",
               "绿光LED":"(500,560)+(30,60)+(0.0,1.0)",
               "蓝光LED":"(445,465)+(15,30)+(0.0,1.0)",
               "色温范围":"(2500,15000)",
               "显色指数":"70",
               "迭代次数":"10000"}
    values = []
    for item in setItem:
        values.append(default[item])
        
    inputItem = multenterbox("输入规范：\n\n"\
                             "输入项：红光LED >>> 输入格式：(615,650)+(10,20)+(0.0,1.0)\n\n"\
                             "输入项：色温范围 >>> 输入格式：(2500,15000)\n\n"\
                             "输入项：显色指数 >>> 输入格式：70\n\n"\
                             "输入项：迭代次数 >>> 输入格式：10000\n\n"\
                             "<<<请根据输入框的格式更改设置>>>"
                             ,"输入设置参数",setItem,values)
    return inputItem


#****************************主函数*********************************
if __name__ == "__main__":

    #***************************GUI输入*******************************
    ynInput = 0
    #*************默认参数*******************
    Rled = {"waveBd":(615,650),"fwhm":(10,20),"ratio":(0,1)}
    Gled = {"waveBd":(500,560),"fwhm":(30,60),"ratio":(0,1)}
    Bled = {"waveBd":(445,465),"fwhm":(15,30),"ratio":(0,1)}
    TempColor = (2500,15000)
    Index_Ra = 70
    iteration = 10000
    #*************默认参数*******************

    while not ynInput:
        setItem = initSet()#选择需要更改的设置项目
        inputItem = initInput(setItem)#输入更改项目的参数
        setValue = dict(list(zip(setItem,inputItem)))#获取输入项目字典
        
        #************参数重写********************
        re1 = compile(r"(\d*(?:\.\d*)?),(\d*(?:\.\d*)?)")
        if "红光LED" in setValue:
            temp = findall(re1,setValue["红光LED"])
            Rled["waveBd"] = (float(temp[0][0]),float(temp[0][1]))
            Rled["fwhm"] = (float(temp[1][0]),float(temp[1][1]))
            Rled["ratio"] = (float(temp[2][0]),float(temp[2][1]))

        if "绿光LED" in setValue:
            temp = findall(re1,setValue["绿光LED"])
            Gled["waveBd"] = (float(temp[0][0]),float(temp[0][1]))
            Gled["fwhm"] = (float(temp[1][0]),float(temp[1][1]))
            Gled["ratio"] = (float(temp[2][0]),float(temp[2][1]))

        if "蓝光LED" in setValue:
            temp = findall(re1,setValue["蓝光LED"])
            Bled["waveBd"] = (float(temp[0][0]),float(temp[0][1]))
            Bled["fwhm"] = (float(temp[1][0]),float(temp[1][1]))
            Bled["ratio"] = (float(temp[2][0]),float(temp[2][1]))

        if "色温范围" in setValue:
            temp = findall(re1,setValue["色温范围"])
            TempColor = (float(temp[0][0]),float(temp[0][1]))

        if "显色指数" in setValue:
            temp = findall(r"(\d*(?:\.\d*)?)",setValue["显色指数"])
            Index_Ra = float(temp[0])

        if "迭代次数" in setValue:
            temp = findall(r"(\d*(?:\.\d*)?)",setValue["迭代次数"])
            iteration = int(float(temp[0]))
        #************参数重写********************

        #确认输入参数
        ynInput = boolbox(msg = ("红光LED >>>\n波长范围:%.1f-%.1f   半高宽范围:%.1f-%.1f   功率占比范围:%.1f-%.1f\n"\
                       "绿光LED >>>\n波长范围:%.1f-%.1f   半高宽范围:%.1f-%.1f   功率占比范围:%.1f-%.1f\n"\
                       "蓝光LED >>>\n波长范围:%.1f-%.1f   半高宽范围:%.1f-%.1f   功率占比范围:%.1f-%.1f\n"\
                       "色温范围:%.1f-%.1f\n"\
                       "显色指数:%.1f\n"\
                       "迭代次数:%d"\
                       %(Rled["waveBd"][0],Rled["waveBd"][1],Rled["fwhm"][0],Rled["fwhm"][1],Rled["ratio"][0],Rled["ratio"][1],\
                       Gled["waveBd"][0],Gled["waveBd"][1],Gled["fwhm"][0],Gled["fwhm"][1],Gled["ratio"][0],Gled["ratio"][1],\
                       Bled["waveBd"][0],Bled["waveBd"][1],Bled["fwhm"][0],Bled["fwhm"][1],Bled["ratio"][0],Bled["ratio"][1],\
                       TempColor[0],TempColor[1],Index_Ra,iteration)),\
                       title = "参数确认",choices=('确认', '输入错误，重新输入'))        
    #***************************GUI输入*******************************
    print("程序正在运行，请等待.....\n")
    best = {} #存储最佳参数
    best["efflumin"] = 200 #光效控制条件初始化


    # 迭代寻找配色最优解
    for i in range(iteration): #迭代条件
        
        if (i+1)%1000 == 0:
            print("当前迭代次数：%d\n" %(i+1))
            
        # 功率比之和为1
        # 随机生成组分比 功率占比控制条件
        flag_ratio = 1
        while flag_ratio:
            Rratio = rd.normal(0.33,0.5)
            while not Rled["ratio"][0]<Rratio<Rled["ratio"][1]:
                Rratio = rd.normal(0.33,0.5)

            Gratio = rd.normal(0.33,0.5)
            while not Gled["ratio"][0]<Gratio<min(Gled["ratio"][1],1-Rratio):
                Gratio = rd.normal(0.33,0.5)

            Bratio = 1-Rratio-Gratio
            if Bled["ratio"][0]<Bratio<Bled["ratio"][1]:
                flag_ratio = 0

           
        #随机生成峰值波长  波长范围控制条件
        Rwl = rd.uniform(Rled["waveBd"][0],Rled["waveBd"][1])
        Gwl = rd.uniform(Gled["waveBd"][0],Gled["waveBd"][1])
        Bwl = rd.uniform(Bled["waveBd"][0],Bled["waveBd"][1])

        #随机生成半高宽   半高宽控制条件
        Rband = rd.uniform(Rled["fwhm"][0],Rled["fwhm"][1])
        Gband = rd.uniform(Gled["fwhm"][0],Gled["fwhm"][1])
        Bband = rd.uniform(Bled["fwhm"][0],Bled["fwhm"][1])
            
        RGB_ratio = (Rratio,Gratio,Bratio)
        RGB_info = {"blue":(Bwl,Bband),"green":(Gwl,Gband),"red":(Rwl,Rband)}

        #实例化
        colormix = ColorMix(rgb_ratio = RGB_ratio,rgb_info = RGB_info)
        colormix.ColorTemp()
        Tc = colormix.ColorTemp()["colorTemp"]

        #先检查Tc是否满足要求，如果不满足，则跳出  色温控制条件
        if (not colormix.flag_Tc) or (not TempColor[0]<=Tc<=TempColor[1]):
            continue
        
        Ra = colormix.Aberration()
        efflumin = colormix.eff_lumin()["luminEff"]
        #色温和光效控制条件
        if Ra >= Index_Ra and efflumin > best["efflumin"]:
            best["efflumin"] = efflumin
            best["ratio"] = RGB_ratio
            best["wl"] = (Rwl,Gwl,Bwl)
            best["band"] = (Rband,Gband,Bband)


    # 原来出现的一个问题就是这里的半高宽跟上面设置不一致，导致错误总是检查不出来
    RGB_info = {"blue":(best["wl"][2],best["band"][2]),\
                "green":(best["wl"][1],best["band"][1]),"red":(best["wl"][0],best["band"][0])}    
    colormix = ColorMix(rgb_ratio = best["ratio"],rgb_info = RGB_info)

    '''
    #测试用：结果：色温8695，显色指数83.8    
    RGB_info = {"blue":(462,20),"green":(551,50),"red":(620,15)}
    colormix = ColorMix(rgb_ratio =(1.17,0.63,1.4),rgb_info = RGB_info)
    '''
    Tc = colormix.ColorTemp()["colorTemp"]
    Ra = colormix.Aberration()

    
    # 结果输出
    print("迭代次数：%d\n"%iteration)
    print("RGB三原色的波长：【%f nm】【%f nm】【%f nm】\n"%(best["wl"]))
    print("RGB三原色的半高宽：【%f nm】【%f nm】【%f nm】\n"%(best["band"]))

    effLumin = colormix.eff_lumin()

    print("待测光源的总功率：%f W\n"%effLumin["power"])
    print("待测光源的流明数：%f lm\n"%effLumin["lumin"])
    print("待测光源的视觉效率：%f lm/W\n"%effLumin["luminEff"])

    colormix.rgbSpec()
    spec = colormix.rgbSpec()
    print("待测光源的RGB三色功率之比：%f:%f:%f\n"%spec["rgbPowerRatio"])
    
    
    print("待测光源的色温：%f K\n" %Tc)
    print("待测光源的显色指数：%f\n" %Ra)
    
    colormix.flag_rgb = True #画图旗帜
    colormix.blackbodyTemp = Tc
    colormix.rgbSpec()


    
    
