/*
 * (C) 2007-2010 TaoBao Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * ob_client_helper.cpp is for what ...
 *
 * Version: $id$
 *
 * Authors:
 *   MaoQi maoqi@taobao.com
 *
 */

#include "ob_client_helper.h"
#include "common/ob_client_manager.h"
#include "common/ob_result.h"
#include "common/utility.h"
#include "common/thread_buffer.h"

namespace oceanbase
{
  namespace common
  {
    int ObClientHelper2::parse_merge_server(ObServer* merge_servers, ObScanner& scanner)
    {
      ObServer server;
      ObString start_key;
      ObString end_key; 
      ObCellInfo * cell = NULL;
      ObScannerIterator iter; 
      bool row_change = false;
      int index = 0;
      int ret = OB_SUCCESS;

      int64_t ip = 0;
      int64_t port = 0;
      int64_t version = 0;
      iter = scanner.begin();
      ret = iter.get_cell(&cell, &row_change);
      row_change = false;

      while((OB_SUCCESS == ret))
      {
        if (ret != OB_SUCCESS)
        {
          TBSYS_LOG(ERROR, "get cell from scanner iterator failed:ret[%d]", ret);
        }
        else if (row_change && index > 0)
        {
          TBSYS_LOG(DEBUG,"row changed,ignore"); 
          hex_dump(cell->row_key_.ptr(),cell->row_key_.length(),false,TBSYS_LOG_LEVEL_DEBUG);
          break; //just get one row        
        } 
        else if (cell != NULL)
        {
          end_key.assign(cell->row_key_.ptr(), cell->row_key_.length());
          if ((cell->column_name_.compare("1_ms_port") == 0) 
              || (cell->column_name_.compare("2_ms_port") == 0) 
              || (cell->column_name_.compare("3_ms_port") == 0))
          {
            ret = cell->value_.get_int(port);
            TBSYS_LOG(DEBUG,"port is %ld",port);
          }
          else if ((cell->column_name_.compare("1_ipv4") == 0)
              || (cell->column_name_.compare("2_ipv4") == 0)
              || (cell->column_name_.compare("3_ipv4") == 0))
          {
            ret = cell->value_.get_int(ip);
            TBSYS_LOG(DEBUG,"ip is %ld",ip);
          }
          else if (cell->column_name_.compare("1_tablet_version") == 0 ||
              cell->column_name_.compare("2_tablet_version") == 0 ||
              cell->column_name_.compare("3_tablet_version") == 0)
          {
            ret = cell->value_.get_int(version);
            hex_dump(cell->row_key_.ptr(),cell->row_key_.length(),false,TBSYS_LOG_LEVEL_DEBUG);
            TBSYS_LOG(DEBUG,"tablet_version is %d",version);
          }

          if (OB_SUCCESS == ret)
          {
            if (0 != port && 0 != ip && 0 != version)
            {
              TBSYS_LOG(DEBUG,"ip,port,version:%ld,%ld,%d",ip,port,version);
              merge_server[index++].set_ipv4_addr(ip, port);
              ip = port = version = 0;
            }
          }
          else 
          {
            TBSYS_LOG(ERROR, "check get value failed:ret[%d]", ret);
          }

          if (++iter == scanner.end())
            break;
          ret = iter.get_cell(&cell, &row_change);
        }
        else
        {
          //impossible
        }
      }
      return ret;
    }

    ObClientHelper2::ObClientHelper2() :inited_(false),client_manager_(NULL),thread_buffer_(NULL),timeout_(100*1000L)
    {}

    void ObClientHelper2::init(ObClientManager* client_manager, ThreadSpecificBuffer *thread_buffer,
                              const ObServer root_server, int64_t timeout)
    {
      if (!inited_)
      {
        client_manager_ = client_manager;
        thread_buffer_ = thread_buffer;
        root_server_ = root_server;
        timeout_ = timeout;
        inited_ = true;
      }
    }
    
    int ObClientHelper2::scan(const ObScanParam& scan_param,ObScanner& scanner)
    {
      int ret = OB_SUCCESS;

      if (!inited_)
      {
        ret = OB_NOT_INIT; 
      }

      if ((OB_SUCCESS == ret) && 
          ((ret = get_tablet_info(scan_param)) != OB_SUCCESS) )
      {
        TBSYS_LOG(ERROR,"cann't get mergeserver,ret = %d",ret);
      }

      for(uint32_t i=0; i < sizeof(merge_server_) / sizeof(merge_server_[0]); ++i)
      {
        if ( 0 != merge_server_[i].get_ipv4() && 0 != merge_server_[i].get_port())
        {
          if ((ret = scan(merge_server_[i],scan_param,scanner)) != OB_SUCCESS)
          {
            char tmp_buf[32];
            merge_server_[i].to_string(tmp_buf,sizeof(tmp_buf));
            TBSYS_LOG(INFO,"scan from (%s)",tmp_buf);
            if (OB_RESPONSE_TIME_OUT == ret || OB_PACKET_NOT_SENT == ret)
            {
              TBSYS_LOG(WARN,"scan from (%s) error,ret = %d",tmp_buf,ret);
              continue; //retry
            }
          }
          break;
        }
      }
      return ret;
    }
    
    int ObClientHelper2::get(const ObGetParam& get_param,ObScanner& scanner)
    {
      int ret = OB_SUCCESS;

      if (!inited_)
      {
        ret = OB_NOT_INIT; 
      }

      if ((OB_SUCCESS == ret) && 
          ((ret = get_tablet_info(get_param)) != OB_SUCCESS) )
      {
        TBSYS_LOG(ERROR,"cann't get mergeserver,ret = %d",ret);
      }

      for(uint32_t i=0; i < sizeof(merge_server_) / sizeof(merge_server_[0]); ++i)
      {
        if ( 0 != merge_server_[i].get_ipv4() && 0 != merge_server_[i].get_port())
        {
          char tmp_buf[32];
          merge_server_[i].to_string(tmp_buf,sizeof(tmp_buf));
          TBSYS_LOG(DEBUG,"get from (%s)",tmp_buf);
        
          if ((ret = get(merge_server_[i],get_param,scanner)) != OB_SUCCESS)
          {
            if (OB_RESPONSE_TIME_OUT == ret || OB_PACKET_NOT_SENT == ret)
            {
              TBSYS_LOG(WARN,"get from (%s) error,ret = %d",tmp_buf,ret);
              continue; //retry
            }
          }
          break;
        }
      }
      return ret;
    }


    int ObClientHelper2::scan(const ObServer& server,const ObScanParam& scan_param,ObScanner& scanner)
    {
      int ret = OB_SUCCESS;
      static const int MY_VERSION = 1;
      
      if (!inited_)
      {
        ret = OB_NOT_INIT; 
      }

      if (OB_SUCCESS == ret)
      {
        ObDataBuffer data_buff;
        get_thread_buffer_(data_buff);
        ret = scan_param.serialize(data_buff.get_data(), data_buff.get_capacity(), data_buff.get_position());

        if (OB_SUCCESS == ret)
        {
          ret = client_manager_->send_request(server, OB_SCAN_REQUEST, MY_VERSION, timeout_, data_buff);
          if (OB_SUCCESS != ret)
          {
            char tmp_buf[32];
            server.to_string(tmp_buf,sizeof(tmp_buf));
            TBSYS_LOG(WARN, "failed to send request to (%s), ret=%d", tmp_buf,ret);
          }
        }

        if (OB_SUCCESS == ret)
        {
          // deserialize the response code
          int64_t pos = 0;
          if (OB_SUCCESS == ret)
          {
            ObResultCode result_code;
            ret = result_code.deserialize(data_buff.get_data(), data_buff.get_position(), pos);
            if (OB_SUCCESS != ret)
            {
              TBSYS_LOG(ERROR, "deserialize result_code failed:pos[%ld], ret[%d]", pos, ret);
            }
            else
            {
              ret = result_code.result_code_;
              if (OB_SUCCESS == ret
                  && OB_SUCCESS != (ret = scanner.deserialize(data_buff.get_data(), data_buff.get_position(), pos)))
              {
                TBSYS_LOG(ERROR, "deserialize result data failed:pos[%ld], ret[%d]", pos, ret);
              }
            }
          }
        }
      }
      return ret;
    }
    
    int ObClientHelper2::get(const ObServer& server,const ObGetParam& get_param,ObScanner& scanner)
    {
      int ret = OB_SUCCESS;
      static const int MY_VERSION = 1;
      
      if (!inited_)
      {
        ret = OB_NOT_INIT; 
      }

      if (OB_SUCCESS == ret)
      {
        ObDataBuffer data_buff;
        get_thread_buffer_(data_buff);
        ret = get_param.serialize(data_buff.get_data(), data_buff.get_capacity(), data_buff.get_position());

        if (OB_SUCCESS == ret)
        {
          ret = client_manager_->send_request(server, OB_GET_REQUEST, MY_VERSION, timeout_, data_buff);
          if (OB_SUCCESS != ret)
          {
            char tmp_buf[32];
            server.to_string(tmp_buf,sizeof(tmp_buf));
            TBSYS_LOG(WARN, "failed to send request to (%s), ret=%d", tmp_buf,ret);
          }
        }

        if (OB_SUCCESS == ret)
        {
          // deserialize the response code
          int64_t pos = 0;
          if (OB_SUCCESS == ret)
          {
            ObResultCode result_code;
            ret = result_code.deserialize(data_buff.get_data(), data_buff.get_position(), pos);
            if (OB_SUCCESS != ret)
            {
              TBSYS_LOG(ERROR, "deserialize result_code failed:pos[%ld], ret[%d]", pos, ret);
            }
            else
            {
              ret = result_code.result_code_;
              if (OB_SUCCESS == ret
                  && OB_SUCCESS != (ret = scanner.deserialize(data_buff.get_data(), data_buff.get_position(), pos)))
              {
                TBSYS_LOG(ERROR, "deserialize result data failed:pos[%ld], ret[%d]", pos, ret);
              }
            }
          }
        }
      }
      return ret;
    }
  
    int ObClientHelper2::apply(const ObServer& update_server, const ObMutator& mutator)
    {
      int ret = OB_SUCCESS;
      static const int MY_VERSION = 1;

      if (!inited_)
      {
        ret = OB_NOT_INIT; 
      }

      if (OB_SUCCESS == ret)
      {
        ObDataBuffer data_buff;
        get_thread_buffer_(data_buff);
        ret = mutator.serialize(data_buff.get_data(), data_buff.get_capacity(), data_buff.get_position());

        if (OB_SUCCESS == ret)
        {
          ret = client_manager_->send_request(update_server, OB_WRITE, MY_VERSION, timeout_, data_buff);
          if (OB_SUCCESS != ret)
          {
            char tmp_buf[32];
            update_server.to_string(tmp_buf,sizeof(tmp_buf));
            TBSYS_LOG(WARN, "failed to send request to (%s), ret=%d",tmp_buf,ret);
          }
        }

        if (OB_SUCCESS == ret)
        {
          // deserialize the response code
          int64_t pos = 0;
          if (OB_SUCCESS == ret)
          {
            ObResultCode result_code;
            ret = result_code.deserialize(data_buff.get_data(), data_buff.get_position(), pos);
            if (OB_SUCCESS != ret)
            {
              TBSYS_LOG(ERROR, "deserialize result_code failed:pos[%ld], ret[%d]", pos, ret);
            }
            else if (result_code.result_code_ != OB_SUCCESS)
            {
              TBSYS_LOG(ERROR,"apply failed : %d",result_code.result_code_);
            }
          }
        }
      }
      return ret;
    }

    
    int ObClientHelper2::get_tablet_info(const ObScanParam& scan_param)
    {
      ObScanner scanner;
      int err = OB_SUCCESS;

      if (OB_SUCCESS != (err = scan(root_server_,scan_param,scanner)))
      {
        TBSYS_LOG(ERROR,"get tablet from rootserver(%s) failed:[%d]", , err);
      }

      if (OB_SUCCESS == ret)
      {
        ret = parse_merge_server(scanner); 
      }
      return  ret;
    }
    
    int ObClientHelper2::get_tablet_info(const ObGetParam& param)
    {
      int ret = OB_SUCCESS;
      ObScanner scanner;

      if ((ret = get(root_server_,param,scanner)) != OB_SUCCESS) 
      {
        char tmp_buf[32];
        root_server_.to_string(tmp_buf,sizeof(tmp_buf));
        TBSYS_LOG(ERROR,"get tablet from rootserver(%s) failed:[%d]",tmp_buf,ret);
      }

      if (OB_SUCCESS == ret)
      {
        ret = parse_merge_server(scanner);
      }

      return ret;
    }


    int ObClientHelper2::get_thread_buffer_(ObDataBuffer& data_buff)
    {
      int err = OB_SUCCESS;
      ThreadSpecificBuffer::Buffer* buffer = thread_buffer_->get_buffer();
      buffer->reset();
      data_buff.set_data(buffer->current(), buffer->remain());
      return err;
    }
    
  } /* common */
  
} /* oceanbase */
