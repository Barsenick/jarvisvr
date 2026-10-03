using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using JarvisVR.Net;

namespace JarvisVR.Audio
{
    /// <summary>
    /// Captures audio from the Quest microphone and streams it as raw PCM16 
    /// binary frames to the JarvisConnection bridge.
    /// </summary>
    [RequireComponent(typeof(JarvisConnection))]
    public class JarvisMicStreamer : MonoBehaviour
    {
        [SerializeField] private int sampleRate = 16000;
        [SerializeField] private int frameSize = 480; // 30ms at 16kHz
        
        private JarvisConnection _connection;
        private AudioClip _micClip;
        private int _lastSamplePos = 0;
        private bool _isStreaming = false;

        private void Start()
        {
            _connection = GetComponent<JarvisConnection>();
            StartCoroutine(InitMic());
        }

        private IEnumerator InitMic()
        {
            // Wait for the mic to initialize
            while (Microphone.devices.Length == 0)
            {
                Debug.LogWarning("[MicStreamer] No microphone detected... waiting.");
                yield return new WaitForSeconds(1f);
            }

            string device = Microphone.devices[0];
            _micClip = Microphone.Start(device, true, 1, sampleRate);
            
            Debug.Log($"[MicStreamer] Started recording from {device} at {sampleRate}Hz");
            _isStreaming = true;
            StartCoroutine(StreamLoop());
        }

        private IEnumerator StreamLoop()
        {
            while (true)
            {
                if (_isStreaming && _connection != null && _connection.IsReady)
                {
                    int pos = Microphone.GetPosition(null);
                    int diff = pos - _lastSamplePos;

                    if (diff >= frameSize)
                    {
                        float[] samples = new float[diff];
                        _micClip.GetData(samples, _lastSamplePos);
                        
                        byte[] pcm16 = ConvertToPcm16(samples);
                        _connection.SendRaw(pcm16).Forget(); // Fire and forget
                        
                        _lastSamplePos = pos;
                    }
                    
                    if (pos < _lastSamplePos) _lastSamplePos = 0; // Reset on wrap
                }
                yield return new WaitForSeconds(0.02f); // ~20ms check
            }
        }

        private byte[] ConvertToPcm16(float[] samples)
        {
            byte[] pcm = new byte[samples.Length * 2];
            for (int i = 0; i < samples.Length; i++)
            {
                // Clamp and scale to Int16
                short s = (short)(Mathf.Clamp(samples[i], -1f, 1f) * 32767);
                byte[] bytes = BitConverter.GetBytes(s);
                pcm[i * 2] = bytes[0];
                pcm[i * 2 + 1] = bytes[1];
            }
            return pcm;
        }
    }

    public static class TaskExtensions
    {
        public static async void Forget(this System.Threading.Tasks.Task task)
        {
            try { await task; } catch (Exception e) { Debug.LogException(e); }
        }
    }
}
