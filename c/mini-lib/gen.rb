#!/usr/bin/env ruby

module Gen
require 'erb'

class TaggedFile
    attr_reader :file_name
    def initialize(file_name)
        @file_name = file_name
        text = File.open(@file_name).read()
        text.scan /^\/\/<<<(\w+)(?:\|(\w+))?(.+?)^\/\/(\w+)>>>/m do |m|
            name, method, str, endname= m
            raise "no matching tag!" if name != endname
            self.instance_eval "@#{name} = str "
            self.instance_eval "@#{name} = #{method}(@#{name})" if method
            self.class.class_eval "attr_reader :#{name}"
        end
    end
end

class TaggedCFile <TaggedFile
    def get_func_list(text)
        text.scan(/^\w[^{]+$/).find_all { |m| m !~ /^static/ }
    end
end 

module FileUpdate
    def need_update(file_name, buf)
        File.exist?(file_name) ? (File.open(file_name).read() != buf) : true
    end

    def update_file(file_name, buf)
         if need_update(file_name, buf)
             file = File.open(file_name, 'w')
             file.write(buf)
             file.close()
         end
    end
end


class Header < ERB
    include FileUpdate
    def initialize(tagged_file)
        header_template = <<-endheader
#ifndef <%=header_guard%>
#define <%=header_guard%>
<%=header%>

% func_list.each do |func|
<%=func%>;
% end

#endif /* <%=header_guard%> */
endheader
        super(header_template, 0, "%<>")
        @file_name = tagged_file.file_name.gsub(/\.c$/,'.h')
        header_guard = "_#{@file_name.gsub('.', '_').upcase}_"
        header = tagged_file.header
        func_list = tagged_file.func_list
        update_file(@file_name, result(binding))
    end
end

def Gen.gen_header(src)
    Header.new(TaggedCFile.new(src))
end

def Gen.get_func_list_from_binary_file(file)
    buf=`objdump -t #{file}`
    buf.scan(/F .text\s+[0-9a-f]+\s+(\w+)$/).flatten
end

#eval("#{ARGV[0]}(*ARGV[1..ARGV.length])")
end
