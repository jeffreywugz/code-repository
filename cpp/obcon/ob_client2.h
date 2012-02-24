#include "common/ob_define.h"
#include "common/ob_result.h"
#include "common/ob_packet_factory.h"
#include "common/ob_client_manager.h"

using namespace oceanbase::common;

struct Dummy
{
  int serialize(char* buf, int64_t len, int64_t& pos) const
  {
    return OB_SUCCESS;
  }
  int deserialize(char* buf, int64_t len, int64_t& pos)
  {
    return OB_SUCCESS;
  }
};

class BaseClient
{
  public:
    BaseClient()
  {
  }
    virtual ~BaseClient()
  {
  }
  public:
    virtual int initialize()
  {
    streamer_.setPacketFactory(&factory_);
    client_.initialize(&transport_, &streamer_);
    return transport_.start();
  }

    virtual int destroy()
  {
    transport_.stop();
    return transport_.wait();
  }

    virtual int wait()
  {
    return transport_.wait();
  }

    ObClientManager * get_rpc()
  {
    return &client_;
  }

  public:
    tbnet::DefaultPacketStreamer streamer_;
    tbnet::Transport transport_;
    ObPacketFactory factory_;
    ObClientManager client_;
};

template <class T>
int uni_serialize(const T &data, char *buf, const int64_t data_len, int64_t& pos)
{
  return data.serialize(buf, data_len, pos);
};

template <class T>
int uni_deserialize(T &data, char *buf, const int64_t data_len, int64_t& pos)
{
  return data.deserialize(buf, data_len, pos);
};

template <>
int uni_serialize<uint64_t>(const uint64_t &data, char *buf, const int64_t data_len, int64_t& pos)
{
  return serialization::encode_vi64(buf, data_len, pos, (int64_t)data);
};

template <>
int uni_serialize<int64_t>(const int64_t &data, char *buf, const int64_t data_len, int64_t& pos)
{
  return serialization::encode_vi64(buf, data_len, pos, data);
};

template <>
int uni_deserialize<uint64_t>(uint64_t &data, char *buf, const int64_t data_len, int64_t& pos)
{
  return serialization::decode_vi64(buf, data_len, pos, (int64_t*)&data);
};

template <>
int uni_deserialize<int64_t>(int64_t &data, char *buf, const int64_t data_len, int64_t& pos)
{
  return serialization::decode_vi64(buf, data_len, pos, &data);
};

class ObBaseMockClient : public BaseClient
{
  private:
    static const int64_t BUF_SIZE = 2 * 1024 * 1024;

  public:
    ObBaseMockClient()
  {}

    ~ObBaseMockClient()
  {}

    int init(const ObServer& server, int64_t timeout)
  {
    int err = OB_SUCCESS;

    timeout_ = timeout;
    initialize();
    server_ = server;

    return err;
  }

  public:
    template <class Input, class Output>
    int send_request(const int pcode, const Input &param, Output &result, const int64_t timeout=1000*1000)
  {
    int ret = OB_SUCCESS;
    static const int32_t MY_VERSION = 1;
    ObDataBuffer data_buff;
    get_thread_buffer_(data_buff);

    ObClientManager* client_mgr = get_rpc();
    ret = uni_serialize(param, data_buff.get_data(), data_buff.get_capacity(), data_buff.get_position());
    if (OB_SUCCESS == ret)
    {
      ret = client_mgr->send_request(server_, pcode, MY_VERSION, timeout, data_buff);
      if (OB_SUCCESS != ret)
      {
        TBSYS_LOG(WARN, "failed to send request, ret=%d", ret);
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
              && OB_SUCCESS != (ret = uni_deserialize(result, data_buff.get_data(), data_buff.get_position(), pos)))
          {
            TBSYS_LOG(ERROR, "deserialize result data failed:pos[%ld], ret[%d]", pos, ret);
          }
        }
      }
    }
    return ret;
  }
    
  //   int fetch_schema(ObSchemaManagerV2& schema_mgr)
  // {
  //   return send_request(OB_FETCH_SCHEMA, dummy_, schema_mgr, timeout_);
  // }

    ThreadSpecificBuffer& get_rpc_buffer()
  {
    return rpc_buffer_;
  }
    int get_thread_buffer_(ObDataBuffer& data_buff) const
    {
      int err = OB_SUCCESS;
      ThreadSpecificBuffer::Buffer* buffer = rpc_buffer_.get_buffer();

      buffer->reset();
      data_buff.set_data(buffer->current(), buffer->remain());

      return err;
    }

    Dummy dummy_;    
    ObServer server_;
    ThreadSpecificBuffer rpc_buffer_;
    int64_t timeout_;    
};
