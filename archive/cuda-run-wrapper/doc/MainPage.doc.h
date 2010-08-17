/**
 * @mainpage meta info
 * 
 * @section description cuda run wrapper
 * 这个软件包的目的是使在cuda平台下，多个程序使用多个GPU变得更为容易。使用原始的cuda runtime API进行编程，
 * 当一个cuda程序运行在多GPU环境下时， 程序员必须显式的指明自己要使用哪个GPU，否则将选择0号GPU。显然，这样
 * 如果没有程序员的手工干预，就完全无法实现多个GPU之间的负载均衡，即使程序员手工指定GPU的分配, 由于程序员
 * 不可能知道运行时的全局信息，也不可能正真实现负载均衡。cuda-run-wrapper的目的就是提供一个启动
 * cuda程序的封装层，使得如果通过此封装层启动cuda程序的话，就可以实现即不需要程序员显示指定cuda程序将要使用那个GPU，
 * 又能尽可能的达到负载均衡。
 * 
 * 项目主页: <a href="http://code.google.com/p/cuda-run-wrapper">http://code.google.com/p/cuda-run-wrapper</a> \n
 * @section LICENSE
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
 * @date June 2009
 * @version 0.1
 * @author xihuafeng <a href="mailto:huafengxi@gmail.com">huafengxi@gmail.com</a>
 * 
 */

/**
 * @defgroup daemon cuda run wrapper daemon
 * 
 */

/**
 * @defgroup libcudarun cuda run wrapper shared library
 * 
 */

/**
 * @defgroup device cuda device abstract
 * 
 */

/**
 * @defgroup libphi mini lib write for this project
 * 
 */
